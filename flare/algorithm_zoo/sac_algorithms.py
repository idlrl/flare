import torch
import torch.nn.functional as F
from flare.algorithm_zoo.simple_algorithms import SimpleAC
from flare.framework import common_functions as comf


class SAC(SimpleAC):
    """
    Simple implementation of Soft Actor-Critic.

    "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement
    Learning with a Stochastic Actor"
    http://proceedings.mlr.press/v80/haarnoja18b/haarnoja18b.pdf

    learn() requires keywords: "action", "reward", "v_value", "q_value"
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 value_cost_weight=0.5,
                 prob_entropy_weight=0.01,
                 lambda_Q=0.1,
                 lambda_Pi=0.1):
        super(SAC, self).__init__(model, gpu_id, discount_factor,
                                  value_cost_weight, prob_entropy_weight)
        self.lambda_V = value_cost_weight
        self.lambda_Q = lambda_Q
        self.lambda_Pi = lambda_Pi

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):

        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        V = values["v_value"]
        Q = values["q_value"]

        with torch.no_grad():
            next_values, next_states_update = self.model.value(next_inputs,
                                                               next_states)
            V_hat = next_values["v_value"] * torch.abs(next_alive["alive"])
        assert V.size() == V_hat.size()

        dist, _ = self.model.policy(inputs, states)
        pi = dist["action"]
        pi_dist = pi.probs
        log_pi = pi_dist.log()

        # J_V
        target_V = Q - log_pi
        expected_target_V = torch.matmul(
            pi_dist.unsqueeze(1), target_V.unsqueeze(2)).squeeze(-1)
        V_diff = V - expected_target_V
        J_V = 0.5 * (V_diff**2)

        # J_Q
        Q_hat = reward + self.discount_factor * V_hat
        Q_i = comf.idx_select(Q, action)
        Q_diff = Q_i - Q_hat
        J_Q = 0.5 * (Q_diff**2)

        # J_Pi
        target = F.softmax(Q, 1)
        J_pi = F.kl_div(log_pi, target, reduction="none").sum(-1, keepdim=True)

        cost = self.lambda_V * J_V + self.lambda_Q * J_Q + self.lambda_Pi * J_pi

        avg_cost = comf.get_avg_cost(cost)
        avg_cost.backward(retain_graph=True)
        return dict(cost=cost), states_update, next_states_update
