from abc import ABCMeta, abstractmethod
from copy import deepcopy
import numpy as np
from torch.distributions import Categorical
import torch
import torch.optim as optim
from torch.optim import Optimizer
from flare.framework.algorithm import Algorithm
from flare.framework import common_functions as comf
import flare.framework.recurrent as rc


class SimpleAlgorithm(Algorithm):
    """
    A base class for simple algorithms in this file.
    """
    __metaclass__ = ABCMeta

    def __init__(self, model, gpu_id, discount_factor, optim, grad_clip, ntd):
        super(SimpleAlgorithm, self).__init__(model, gpu_id)
        self.discount_factor = discount_factor
        if isinstance(optim, tuple):
            self.optim = optim[0](self.model.parameters(), **optim[1])
        elif isinstance(optim, Optimizer):
            self.optim = optim
        else:
            assert False, "Incorrect optim type"
        self.ntd = ntd
        self.grad_clip = grad_clip
        self.recurrent_helper = rc.AgentRecurrentHelper()

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):
        self.model.train()
        self.optim.zero_grad()
        if states:
            ## next_values will preserve the sequential information!
            next_values = self.recurrent_helper.recurrent(
                ## step function operates one-level lower
                recurrent_step=self.compute_next_values,
                input_dict_list=[next_inputs, next_actions, next_alive],
                state_dict_list=[next_states])

            if self.ntd:  ## we need sequential information for n-step TD
                rewards = {k : comf.prepare_ntd_reward(r, self.discount_factor) \
                           for k, r in rewards.items()}
                next_values = {k : comf.prepare_ntd_value(v, self.discount_factor) \
                               for k, v in next_values.items()}

            ## costs will preserve the sequential information!
            costs = self.recurrent_helper.recurrent(
                ## step function operates one-level lower
                recurrent_step=self._rl_learn,
                input_dict_list=[inputs, actions, next_values, rewards],
                state_dict_list=[states])
        else:
            ## If no sequential data, ntd=True will be ignored
            next_values, _ = self.compute_next_values(
                next_inputs, next_actions, next_alive, next_states)
            costs, _ = self._rl_learn(inputs, actions, next_values, rewards,
                                      states)

        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip)
        self.optim.step()
        return costs

    def predict(self, inputs, states):
        self.model.eval()
        return self._rl_predict(self.model, inputs, states)

    @abstractmethod
    def compute_next_values(self, next_inputs, next_actions, next_alive,
                            next_states):
        """
        A child class must implement this to decide how to compute next values

        Return: next_values(dict), next_states_update(dict)
        """
        pass

    @abstractmethod
    def _rl_learn(self, inputs, actions, next_values, rewards, states):
        """
        A child class must implement this learning function to compute costs and gradients

        Return: costs(dict), states_update(dict)
        """
        pass


class SimpleAC(SimpleAlgorithm):
    """
    A simple Actor-Critic that has a feedforward policy network and
    a single action.
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 value_cost_weight=1.0,
                 prob_entropy_weight=0.01,
                 optim=(optim.RMSprop, dict(lr=1e-4)),
                 grad_clip=None,
                 ntd=False):

        super(SimpleAC, self).__init__(model, gpu_id, discount_factor, optim,
                                       grad_clip, ntd)
        self.value_cost_weight = value_cost_weight
        self.prob_entropy_weight = prob_entropy_weight

    def _rl_learn(self, inputs, actions, next_values, rewards, states):
        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        value = values["v_value"]
        next_value = next_values["v_value"]
        assert value.size() == next_value.size()

        critic_value = reward + self.discount_factor * next_value
        td_error = (critic_value - value).squeeze(-1)
        value_cost = td_error**2

        dist, _ = self.model.policy(inputs, states)
        dist = dist["action"]

        if action.dtype == torch.int64 or action.dtype == torch.int32:
            ## for discrete actions, we need to provide scalars to log_prob()
            pg_cost = -dist.log_prob(action.squeeze(-1))
        else:
            pg_cost = -dist.log_prob(action)

        value_cost *= self.value_cost_weight
        pg_cost *= td_error.detach()
        entropy_cost = self.prob_entropy_weight * dist.entropy()

        cost = value_cost + pg_cost - entropy_cost  ## increase entropy for exploration

        sum_cost, _ = comf.sum_cost_tensor(cost)
        sum_cost.backward(retain_graph=True)
        return dict(cost=cost,
                    pg_cost=pg_cost,
                    value_cost=value_cost,
                    entropy_cost=entropy_cost), \
            states_update

    def compute_next_values(self, next_inputs, next_actions, next_alive,
                            next_states):
        with torch.no_grad():
            next_values, next_states_update = self.model.value(next_inputs,
                                                               next_states)
            next_values = {k : v * torch.abs(next_alive["alive"]) \
                           for k, v in next_values.items()}
        return next_values, next_states_update


class SimpleQ(SimpleAlgorithm):
    """
    A simple Q-learning that has a feedforward policy network and a single discrete action.
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 exploration_end_steps=0,
                 exploration_end_rate=0.1,
                 update_ref_interval=100,
                 optim=(optim.RMSprop, dict(lr=1e-4)),
                 grad_clip=None,
                 ntd=False):

        super(SimpleQ, self).__init__(model, gpu_id, discount_factor, optim,
                                      grad_clip, ntd)
        self.update_ref_interval = update_ref_interval
        self.total_batches = 0
        ## create a reference model
        if update_ref_interval:
            self.ref_model = deepcopy(model)
            self.ref_model.to(self.device)
        else:
            self.ref_model = model
        ## setup exploration
        if exploration_end_steps > 0:
            self.exploration_rate = 1.0
            self.exploration_end_rate = exploration_end_rate
            self.exploration_rate_delta \
                = (1 - exploration_end_rate) / exploration_end_steps
        else:
            self.exploration_rate = 0.0

    def predict(self, inputs, states):
        """
        Override the base predict() function to put the exploration rate in inputs
        """
        self.model.eval()
        distributions, states = self.model.policy(inputs, states)
        actions = {}
        for key, dist in distributions.items():
            assert isinstance(dist, Categorical)
            if np.random.uniform(0, 1) < self.exploration_rate:
                ## if to explore, we generate a uniform categorical distribution
                ## we don't have to normalize the probs because Categorical will do that inside
                dist = Categorical(torch.ones_like(dist.probs))
            actions[key] = dist.sample().unsqueeze(-1)

        if self.exploration_rate > 0:
            ## decrease the exploration rate by a small delta value
            self.exploration_rate = max(
                self.exploration_rate - self.exploration_rate_delta,
                self.exploration_end_rate)

        return actions, states

    def _rl_learn(self, inputs, actions, next_values, rewards, states):

        if self.update_ref_interval and self.total_batches % self.update_ref_interval == 0:
            ## copy parameters from self.model to self.ref_model
            self.ref_model.load_state_dict(self.model.state_dict())
        self.total_batches += 1

        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        q_value = values["q_value"]
        next_value = next_values["q_value"]

        value = comf.idx_select(q_value, action)
        critic_value = reward + self.discount_factor * next_value
        cost = (critic_value - value)**2

        sum_cost, _ = comf.sum_cost_tensor(cost)
        sum_cost.backward(retain_graph=True)
        return dict(cost=cost), states_update

    def compute_next_values(self, next_inputs, next_actions, next_alive,
                            next_states):
        with torch.no_grad():
            next_values, next_states_update = self.ref_model.value(next_inputs,
                                                                   next_states)
            next_values = {k : (q * torch.abs(next_alive["alive"])).max(-1)[0].unsqueeze(-1) \
                           for k, q in next_values.items()}
        return next_values, next_states_update


class SimpleSARSA(SimpleQ):
    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 epsilon=0.1,
                 optim=(optim.RMSprop, dict(lr=1e-4)),
                 grad_clip=None,
                 ntd=False):

        super(SimpleSARSA, self).__init__(
            model=model,
            gpu_id=gpu_id,
            discount_factor=discount_factor,
            exploration_end_steps=1,
            exploration_end_rate=epsilon,
            update_ref_interval=0,
            optim=optim,
            grad_clip=grad_clip,
            ntd=ntd)

    def compute_next_values(self, next_inputs, next_actions, next_alive,
                            next_states):
        with torch.no_grad():
            next_action = next_actions["action"]
            next_values, next_states_update = self.model.value(next_inputs,
                                                               next_states)
            next_values = {k : comf.idx_select(q * torch.abs(next_alive["alive"]), next_action) \
                           for k, q in next_values.items()}
        return next_values, next_states_update

    def _rl_learn(self, inputs, actions, next_values, rewards, states):
        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        q_value = values["q_value"]
        next_value = next_values["q_value"]

        critic_value = reward + self.discount_factor * next_value
        cost = (critic_value - comf.idx_select(q_value, action))**2

        sum_cost, _ = comf.sum_cost_tensor(cost)
        sum_cost.backward(retain_graph=True)
        return dict(cost=cost), states_update


class OffPolicyAC(SimpleAC):
    """
    Off-policy AC with clipped importance ratio.
    Refer to PPO2 objective for details.

    learn() requires keywords: "action", "reward", "v_value", "action_log_prob"
    """

    def __init__(self,
                 model,
                 epsilon=0.1,
                 gpu_id=-1,
                 discount_factor=0.99,
                 value_cost_weight=1.0,
                 prob_entropy_weight=0.01,
                 optim=(optim.RMSprop, dict(lr=1e-4)),
                 grad_clip=None,
                 ntd=False):

        super(OffPolicyAC, self).__init__(
            model, gpu_id, discount_factor, value_cost_weight,
            prob_entropy_weight, optim, grad_clip, ntd)
        self.epsilon = epsilon

    def get_action_specs(self):
        ### "action_log_prob" is required by the algorithm but not by the model
        return self.model.get_action_specs() + [
            ("action_log_prob", dict(shape=[1]))
        ]

    def _rl_learn(self, inputs, actions, next_values, rewards, states):
        """
        This learn() is expected to be called multiple times on each minibatch
        """

        action = actions["action"]
        log_prob = actions["action_log_prob"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        value = values["v_value"]
        next_value = next_values["v_value"]
        assert value.size() == next_value.size()

        critic_value = reward + self.discount_factor * next_value
        td_error = (critic_value - value).squeeze(-1)
        value_cost = self.value_cost_weight * td_error**2

        dist, _ = self.model.policy(inputs, states)
        dist = dist["action"]

        if action.dtype == torch.int64 or action.dtype == torch.int32:
            ## for discrete actions, we need to provide scalars to log_prob()
            new_log_prob = dist.log_prob(action.squeeze(-1))
        else:
            new_log_prob = dist.log_prob(action)

        ratio = torch.exp(new_log_prob - log_prob.squeeze(-1))
        ### clip pg_cost according to the ratio
        clipped_ratio = torch.clamp(
            ratio, min=1 - self.epsilon, max=1 + self.epsilon)

        pg_obj = torch.min(input=ratio * td_error.detach(),
                           other=clipped_ratio * td_error.detach())
        entropy_cost = self.prob_entropy_weight * dist.entropy()
        cost = value_cost - pg_obj - entropy_cost  ## increase entropy for exploration

        sum_cost, _ = comf.sum_cost_tensor(cost)
        sum_cost.backward(retain_graph=True)
        return dict(cost=cost,
                    pg_obj=pg_obj,
                    value_cost=value_cost,
                    entropy_cost=entropy_cost), \
            states_update
