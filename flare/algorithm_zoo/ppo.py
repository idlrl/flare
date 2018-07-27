from flare.framework.algorithm import Algorithm
from flare.framework import common_functions as comf
from torch.distributions import Categorical
import torch
import numpy as np
from copy import deepcopy


class PPO2(Algorithm):
    """
    PPO2 of the clipped surrogate objective

    learn() requires keywords: "action", "reward", "v_value", "action_log_prob"
    """

    def __init__(self,
                 model,
                 iterations_per_batch,
                 epsilon=0.1,
                 gpu_id=-1,
                 discount_factor=0.99,
                 value_cost_weight=1.0):

        super(PPO2, self).__init__(model, gpu_id, iterations_per_batch)
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.value_cost_weight = value_cost_weight

    def get_action_specs(self):
        ### "action_log_prob" is required by the algorithm but not by the model
        return self.model.get_action_specs() + [
            ("action_log_prob", dict(shape=[1]))
        ]

    def learn(self, inputs, next_inputs, states, next_states, next_episode_end,
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
            next_value = next_values["v_value"] * (
                1 - next_episode_end["episode_end"])

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
