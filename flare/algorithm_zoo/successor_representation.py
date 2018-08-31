from flare.framework.algorithm import Algorithm
from flare.algorithm_zoo.simple_algorithms import SimpleQ
from flare.framework import common_functions as comf
import torch
import numpy as np
from copy import deepcopy


class SuccessorRepresentationQ(SimpleQ):
    """
    A successor representation (SR) implementation with Q values (SARSA).
    For more details, see

    1. https://www.youtube.com/watch?v=OCHwXxSW70o, Tejas Kulkarni
    2. "Successor Features for Transfer in Reinforcement Learning", Barreto et al., 2017
    3. "Visual Semantic Planning using Deep Successor Representations", Zhu el al., 2017
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 exploration_end_steps=0,
                 exploration_end_rate=0.1,
                 reward_cost_weight=1.0):

        super(SuccessorRepresentationQ, self).__init__(
            model=model,
            gpu_id=gpu_id,
            discount_factor=discount_factor,
            exploration_end_steps=exploration_end_steps,
            exploration_end_rate=exploration_end_rate,
            update_ref_interval=0)

        self.reward_cost_weight = reward_cost_weight

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):
        """
        We keep predict() the same with SimpleQ.
        We have to override learn() to implement the learning of SR.
        This function requires four functions implemented by self.model:

        1. self.model.state_embedding() - receives an observation input
                                          and outputs a compact state feature vector
                                          for predicting immediate rewards

        2. self.model.goal()            - outputs a goal vector that has the same
                                          length with the compact state feature vector.
                                          Sometimes, the goal might depend on some inputs.

        3. self.model.sr()              - given the input,
                                          returns a tensor of successor representations, each
                                          one corresponding to an action.
                                          BxAxD where B is the batch size, A is the number of
                                          actions, and D is the dim of state embedding
        """
        action = actions["action"]
        next_action = next_actions["action"]
        reward = rewards["reward"]

        ## 1. learn to predict rewards
        next_state_embedding = self.model.state_embedding(next_inputs)
        # the goal and reward evaluation should be based on the current inputs
        goal = self.model.goal(inputs)
        pred_reward = comf.inner_prod(next_state_embedding, goal)
        reward_cost = (pred_reward - reward)**2 * self.reward_cost_weight

        ## 2. use Bellman equation to learn successor representation
        srs, states_update = self.model.sr(inputs, states)  ## BxAxD
        state_embedding_dim = srs.shape[-1]
        sr = torch.gather(
            input=srs,
            dim=1,
            index=action.unsqueeze(-1).expand(-1, -1, state_embedding_dim))
        sr = sr.squeeze(1)  ## BxD

        with torch.no_grad():
            next_srs, next_states_update = self.model.sr(next_inputs,
                                                         next_states)
            next_sr = torch.gather(
                input=next_srs,
                dim=1,
                index=next_action.unsqueeze(-1).expand(-1, -1,
                                                       state_embedding_dim))
            next_sr = next_sr.squeeze(1) * torch.abs(next_alive["alive"])

        sr_cost = (
            next_state_embedding.detach() + self.discount_factor * next_sr - sr
        )**2
        sr_cost = sr_cost.mean(-1).unsqueeze(-1)

        avg_cost = comf.get_avg_cost(reward_cost + sr_cost)
        avg_cost.backward(retain_graph=True)

        return dict(cost=reward_cost + sr_cost)
