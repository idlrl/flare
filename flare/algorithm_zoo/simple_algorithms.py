from flare.framework.algorithm import Algorithm
from flare.framework import common_functions as comf
from torch.distributions import Categorical
import torch
import numpy as np
from copy import deepcopy


class SimpleAC(Algorithm):
    """
    A simple Actor-Critic that has a feedforward policy network and
    a single discrete action.

    learn() requires keywords: "action", "reward", "v_value"
    """

    def __init__(self, model, gpu_id=-1, discount_factor=0.99):

        super(SimpleAC, self).__init__(model, gpu_id)
        self.discount_factor = discount_factor

    def learn(self, inputs, next_inputs, states, next_states, next_episode_end,
              actions, rewards):

        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        value = values["v_value"]

        with torch.no_grad():
            next_values, next_states_update = self.model.value(next_inputs,
                                                               next_states)
            next_value = next_values["v_value"] * next_episode_end[
                "next_episode_end"]

        assert value.size() == next_value.size()

        critic_value = reward + self.discount_factor * next_value
        td_error = (critic_value - value).squeeze(-1)
        value_cost = td_error**2

        dist, _ = self.model.policy(inputs, states)
        dist = dist["action"]

        if action.dtype == torch.int64 or action.dtype == torch.int32:
            pg_cost = -dist.log_prob(action.squeeze(-1))
        else:
            pg_cost = -dist.log_prob(action)

        cost = value_cost + pg_cost * td_error.detach()

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
        assert update_ref_interval > 0
        self.update_ref_interval = update_ref_interval
        self.total_batches = 0
        ## create a reference model
        self.ref_model = deepcopy(model)
        self.ref_model.to(self.device)
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

    def learn(self, inputs, next_inputs, states, next_states, next_episode_end,
              actions, rewards):

        if self.total_batches % self.update_ref_interval == 0:
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
            next_q_value = next_values["q_value"] * next_episode_end[
                "next_episode_end"]
            next_value, _ = next_q_value.max(-1)
            next_value = next_value.unsqueeze(-1)

        assert q_value.size() == next_q_value.size()

        value = comf.idx_select(q_value, action)
        critic_value = reward + self.discount_factor * next_value
        td_error = (critic_value - value).squeeze(-1)
        cost = td_error**2

        return dict(cost=cost.unsqueeze(-1)), states_update, next_states_update
