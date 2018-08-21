from copy import deepcopy
import numpy as np
from torch.distributions import Categorical
import torch
from flare.framework.algorithm import Algorithm
from flare.framework import common_functions as comf


class SimpleAC(Algorithm):
    """
    A simple Actor-Critic that has a feedforward policy network and
    a single action.

    learn() requires keywords: "action", "reward", "v_value"
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 value_cost_weight=1.0):

        super(SimpleAC, self).__init__(model, gpu_id)
        self.discount_factor = discount_factor
        self.value_cost_weight = value_cost_weight

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):

        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        value = values["v_value"]

        with torch.no_grad():
            next_values, next_states_update = self.model.value(next_inputs,
                                                               next_states)
            next_value = next_values["v_value"] * torch.abs(next_alive[
                "alive"])

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

        cost = self.value_cost_weight * value_cost + pg_cost * td_error.detach(
        )

        return dict(cost=cost.unsqueeze(-1)), states_update, next_states_update

    def predict(self, inputs, states):
        return self._rl_predict(self.model, inputs, states)


class SimpleQ(Algorithm):
    """
    A simple Q-learning that has a feedforward policy network and a single discrete action.

    learn() requires keywords: "action", "reward", "q_value"
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 exploration_end_steps=0,
                 exploration_end_rate=0.1,
                 update_ref_interval=100):

        super(SimpleQ, self).__init__(model, gpu_id)
        self.discount_factor = discount_factor
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
        distributions, states = self.model.policy(inputs, states)
        actions = {}
        for key, dist in distributions.iteritems():
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

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):

        if self.update_ref_interval and self.total_batches % self.update_ref_interval == 0:
            ## copy parameters from self.model to self.ref_model
            self.ref_model.load_state_dict(self.model.state_dict())
        self.total_batches += 1

        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        q_value = values["q_value"]

        with torch.no_grad():
            next_values, next_states_update = self.ref_model.value(next_inputs,
                                                                   next_states)
            next_q_value = next_values["q_value"] * torch.abs(next_alive[
                "alive"])
            next_value, _ = next_q_value.max(-1)
            next_value = next_value.unsqueeze(-1)

        assert q_value.size() == next_q_value.size()

        value = comf.idx_select(q_value, action)
        critic_value = reward + self.discount_factor * next_value
        cost = (critic_value - value)**2

        return dict(cost=cost), states_update, next_states_update


class C51(SimpleQ):
    """
    Categorical Algorithm (C51) based on SimpleQ.
    Refer https://arxiv.org/pdf/1707.06887.pdf for more details.
    "A distributional perspective on reinforcement learning"

    self.model should have members defined in SimpleModelC51 class.
    """
    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 exploration_end_steps=0,
                 exploration_end_rate=0.1,
                 update_ref_interval=100):

        super(C51, self).__init__(model,
                                  gpu_id,
                                  discount_factor,
                                  exploration_end_steps,
                                  exploration_end_rate,
                                  update_ref_interval)
        dead_dist = [0.] * self.model.bins
        dead_dist[0] = 1.
        self.dead_dist = torch.tensor(dead_dist)
        self.float_vmax = torch.FloatTensor([model.vmax])
        self.float_vmin = torch.FloatTensor([model.vmin])

    def select_q_distribution(self, q_distributions, action):
        """
        Select a Q value distribution according to a given action.
        :param q_distributions: Tensor (batch_size x num_actions x num_atoms).
            Q value histogram distribution for each action. The histogram has a
            support of length num_atoms.
        :param action: Tensor (batch_size x 1). Index of actions for each
            sample in a batch.
        :return: Tensor (batch_size x num_atoms). Q value distribution for a
            action.
        """
        one_hot_action = comf.one_hot(action.squeeze(-1), q_distributions.size()[1])
        one_hot_action = one_hot_action.unsqueeze(1)
        q_distribution = torch.matmul(one_hot_action, q_distributions)
        return q_distribution.squeeze(1)

    def backup(self, z, vmax, vmin, delta_z, reward, discount, next_q_distribution):
        """
        Backup sampled reward and reference q value distribution to current q
        value ditribution.
        :param z: Tensor (num_atoms). Atoms.
        :param vmax: float. Maximum value for the distribution.
        :param vmin: float. Minumum value for the distribution.
        :param delta_z: float. size of bin for the distribution.
        :param reward: Tensor (batch_size, 1). Reward function.
        :param discount: float. Discount factor.
        :param next_q_distribution: Tensor (batch_size x num_atoms). Q value
            distribution.
        :return: Tensor (batch_size x num_atoms). Q value distribution.
        """
        ## Compute the projection of Tz onto the zupport z
        Tz = reward + discount * z
        Tz = torch.min(Tz, vmax)
        Tz = torch.max(Tz, vmin)
        b = (Tz - vmin) / delta_z
        l = b.floor()
        u = b.ceil()

        ## Distribute probability of Tz
        wl = u - b
        wu = b - l
        ml = wl * next_q_distribution
        mu = wu * next_q_distribution
        m = torch.zeros(next_q_distribution.size(), dtype=torch.float)
        m = m.scatter_add_(1, l.long(), ml)
        m = m.scatter_add_(1, u.long(), mu)
        return m

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):

        if self.update_ref_interval and self.total_batches % self.update_ref_interval == 0:
            ## copy parameters from self.model to self.ref_model
            self.ref_model.load_state_dict(self.model.state_dict())
        self.total_batches += 1

        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        q_distributions = values["q_value"]

        with torch.no_grad():
            next_values, next_states_update = self.ref_model.value(next_inputs,
                                                                   next_states)
            ## if not alive, Q value is the minimum.
            alpha = torch.abs(next_alive["alive"]).view(-1, 1, 1)
            next_q_distributions = next_values["q_value"] * alpha + self.dead_dist * (1 - alpha)
            next_expected_q_values = self.ref_model.get_expected_q_values(next_q_distributions)
            _, next_action = next_expected_q_values.max(-1)
            next_action = next_action.unsqueeze(-1)

        assert q_distributions.size() == next_q_distributions.size()

        q_distribution = self.select_q_distribution(q_distributions, action)
        next_q_distribution = self.select_q_distribution(next_q_distributions, next_action)

        critic_value = self.backup(
            self.model.atoms,
            self.float_vmax,
            self.float_vmin,
            self.model.delta_z,
            reward, self.discount_factor,
            next_q_distribution)

        ## Cross-entropy loss
        cost = -torch.matmul(
            critic_value.unsqueeze(1),
            q_distribution.log().unsqueeze(-1)).view(-1, 1)

        return dict(cost=cost), states_update, next_states_update


class SimpleSARSA(SimpleQ):
    def __init__(self, model, gpu_id=-1, discount_factor=0.99, epsilon=0.1):

        super(SimpleSARSA, self).__init__(
            model=model,
            gpu_id=gpu_id,
            discount_factor=discount_factor,
            exploration_end_steps=1,
            exploration_end_rate=epsilon,
            update_ref_interval=0)

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):

        action = actions["action"]
        next_action = next_actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        q_value = values["q_value"]

        with torch.no_grad():
            next_values, next_states_update = self.model.value(next_inputs,
                                                               next_states)
            next_value = comf.idx_select(next_values["q_value"], next_action)
            next_value = next_value * torch.abs(next_alive["alive"])

        critic_value = reward + self.discount_factor * next_value
        cost = (critic_value - comf.idx_select(q_value, action))**2

        return dict(cost=cost), states_update, next_states_update


class OffPolicyAC(Algorithm):
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
                 value_cost_weight=1.0):

        super(OffPolicyAC, self).__init__(model, gpu_id)
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.value_cost_weight = value_cost_weight

    def get_action_specs(self):
        ### "action_log_prob" is required by the algorithm but not by the model
        return self.model.get_action_specs() + [
            ("action_log_prob", dict(shape=[1]))
        ]

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):
        """
        This learn() is expected to be called multiple times on each minibatch
        """

        action = actions["action"]
        log_prob = actions["action_log_prob"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        value = values["v_value"]

        with torch.no_grad():
            next_values, next_states_update = self.model.value(next_inputs,
                                                               next_states)
            next_value = next_values["v_value"] * torch.abs(next_alive[
                "alive"])

        assert value.size() == next_value.size()

        critic_value = reward + self.discount_factor * next_value
        td_error = (critic_value - value).squeeze(-1)
        value_cost = td_error**2

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
        cost = self.value_cost_weight * value_cost - pg_obj

        return dict(cost=cost.unsqueeze(-1)), states_update, next_states_update

    def predict(self, inputs, states):
        return self._rl_predict(self.model, inputs, states)
