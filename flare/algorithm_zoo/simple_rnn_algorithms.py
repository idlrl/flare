from copy import deepcopy
import numpy as np
from torch.distributions import Categorical
import torch
import torch.optim as optim
from flare.framework.algorithm import Algorithm
from flare.framework import common_functions as comf
import flare.framework.recurrent as rc


class SimpleRNNAC(Algorithm):
    """
    A simple Actor-Critic that has a recurrent network and
    a single action.

    learn() requires keywords: "action", "reward", "v_value"
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 value_cost_weight=0.5,
                 prob_entropy_weight=0.01,
                 optim=(optim.RMSprop, dict(lr=1e-3)),
                 grad_clip=None):

        super(SimpleRNNAC, self).__init__(model, gpu_id)
        self.discount_factor = discount_factor
        self.value_cost_weight = value_cost_weight
        self.prob_entropy_weight = prob_entropy_weight
        self.optim = optim[0](self.model.parameters(), **optim[1])
        self.grad_clip = grad_clip

    def __learn(self, inputs, next_inputs, states, next_states, next_alive,
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

        cost = self.value_cost_weight * value_cost \
               + pg_cost * td_error.detach() \
               - self.prob_entropy_weight * dist.entropy()  ## increase entropy for exploration
        self.cost_keys = ["cost"]

        avg_cost = comf.get_avg_cost(cost)
        avg_cost.backward(retain_graph=True)

        return dict(cost=cost), states_update, next_states_update

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):
        self.optim.zero_grad()
        assert states
        # rc_out stores all outputs by recurrent_group
        rc_out = rc.recurrent_group(
            inputs=inputs,
            next_inputs=next_inputs,
            states=states,
            next_states=next_states,
            next_alive=next_alive,
            actions=actions,
            next_actions=next_actions,
            rewards=rewards,
            # TODO: if insts is not [], step_func in recurrent group would
            # fail
            insts=[],
            step_func=(
                lambda lens, keys, *args: rc.step_func(self.__learn, lens, keys, *args)
            ))
        # get cost terms from rc_out
        costs = dict(zip(self.cost_keys, rc_out))
        # If gradient clipping is enabled, we should do this before step() and 
        # after backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           self.grad_clip)
        self.optim.step()

        # TODO: why not return [next_]states_update as did in other algorithms
        # this also causes inconsistency between forward network and rnn 
        # network at line 154 of computation_task.py
        return costs

    def predict(self, inputs, states):
        return self._rl_predict(self.model, inputs, states)
