from copy import deepcopy
import math
from flare.framework.algorithm import Algorithm, Model
from flare.framework import common_functions as comf
import torch
import torch.nn as nn
import torch.optim as optim


def fanin_init(m):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
    bound = 1.0 / math.sqrt(fan_in)
    nn.init.uniform_(m.weight, -bound, bound)
    nn.init.uniform_(m.bias, -bound, bound)


def uniform_init(m, bound):
    nn.init.uniform_(m.weight, -bound, bound)
    nn.init.uniform_(m.bias, -bound, bound)


class ContinuousDeterministicModel(Model):
    """
    Model used by DDPG.
    """

    def __init__(self, input_dims, action_dims):
        super(ContinuousDeterministicModel, self).__init__()
        self.input_dims = input_dims
        self.action_dims = action_dims

        self.policy_fc1 = nn.Linear(input_dims, 400)
        self.policy_fc2 = nn.Linear(400, 300)
        self.policy_fc3 = nn.Linear(300, action_dims)
        fanin_init(self.policy_fc1)
        fanin_init(self.policy_fc2)
        uniform_init(self.policy_fc3, 3e-3)

        self.critic_fc1 = nn.Linear(input_dims, 400)
        self.critic_fc2 = nn.Linear(400 + action_dims, 300)
        self.critic_fc3 = nn.Linear(300, 1)
        fanin_init(self.critic_fc1)
        fanin_init(self.critic_fc2)
        uniform_init(self.critic_fc3, 3e-4)

        self.relu = nn.ReLU()

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.input_dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[self.action_dims]))]

    def policy(self, inputs, states):
        #out = self.policy_fc1(inputs.values()[0])
        out = self.policy_fc1(inputs["sensor"])
        out = self.relu(out)
        out = self.policy_fc2(out)
        out = self.relu(out)
        action = self.policy_fc3(out)
        return dict(action=action), states

    def value(self, inputs, states, actions=None):
        #out = self.critic_fc1(inputs.values()[0])
        out = self.critic_fc1(inputs["sensor"])
        h = self.relu(out)
        if actions is None:
            actions, _ = self.policy(inputs, states)
        a = actions["action"]
        out = self.critic_fc2(torch.cat([h, a], dim=-1))
        out = self.relu(out)
        value = self.critic_fc3(out)
        return dict(q_value=value), states

    def policy_optimizer(self, optim_specs):
        """
        Specify the optimizer settings for policy network. `optim_specs` is a
        tuple of two items: `optim_specs[0]` is a optimizer object from
        `torch.optim`, and `optim_specs[1]` is a dict of arguments for
        `optim_specs[0]`.
        """
        return optim_specs[0]([{
            'params': self.policy_fc1.parameters()
        }, {
            'params': self.policy_fc2.parameters()
        }, {
            'params': self.policy_fc3.parameters()
        }], **optim_specs[1])

    def critic_optimizer(self, optim_specs):
        """
        Specify the optimizer settings for critic network. `optim_specs` is a
        tuple of two items: `optim_specs[0]` is a optimizer object from
        `torch.optim`, and `optim_specs[1]` is a dict of arguments for
        `optim_specs[0]`.
        """
        return optim_specs[0]([{
            'params': self.critic_fc1.parameters()
        }, {
            'params': self.critic_fc2.parameters()
        }, {
            'params': self.critic_fc3.parameters()
        }], **optim_specs[1])


class DDPG(Algorithm):
    """
    DDPG impelmentation. Currently no support for short-term memory.
    """

    def __init__(
            self,
            model,
            gpu_id=-1,
            discount_factor=0.99,
            update_ref_interval=100,
            update_weight=1.0,  # update_weight == 1.0 means hard update
            policy_optim=(optim.RMSprop, dict(lr=1e-3)),
            critic_optim=(optim.RMSprop, dict(lr=1e-3))):
        super(DDPG, self).__init__(model, gpu_id)
        self.discount_factor = discount_factor
        self.update_ref_interval = update_ref_interval
        self.update_weight = update_weight
        self.total_batches = 0
        # define optimizers for policy and critic respectively
        self.policy_optim = self.model.policy_optimizer(policy_optim)
        self.critic_optim = self.model.critic_optimizer(critic_optim)
        ## create a reference model
        if update_ref_interval:
            self.ref_model = deepcopy(model)
            self.ref_model.to(self.device)
        else:
            self.ref_model = model

    def __update_model(self, target, source):
        tau = self.update_weight
        for t_param, param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(t_param.data * (1.0 - tau) + param.data * tau)

    def predict(self, inputs, states):
        """
        Override the base predict() function to put the exploration rate in inputs
        """
        self.model.eval()
        return self.model.policy(inputs, states)

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):
        self.model.train()

        if self.update_ref_interval and self.total_batches % self.update_ref_interval == 0:
            ## copy parameters from self.model to self.ref_model
            self.__update_model(self.ref_model, self.model)
        self.total_batches += 1

        reward = rewards["reward"]

        self.critic_optim.zero_grad()
        values, states_update = self.model.value(inputs, states, actions)
        q_value = values["q_value"]

        with torch.no_grad():
            next_values, next_states_update = self.ref_model.value(next_inputs,
                                                                   next_states)
            next_value = next_values["q_value"] * torch.abs(next_alive[
                "alive"])

        assert q_value.size() == next_value.size()

        critic_value = reward + self.discount_factor * next_value
        critic_loss = (q_value - critic_value).squeeze(-1)**2
        sum_critic_loss, _ = comf.sum_cost_tensor(critic_loss)
        sum_critic_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        values2, _ = self.model.value(inputs, states)
        policy_loss = -values2["q_value"].squeeze(-1)
        sum_policy_loss, _ = comf.sum_cost_tensor(policy_loss)
        sum_policy_loss.backward()
        self.policy_optim.step()

        return dict(critic_loss=critic_loss, policy_loss=policy_loss)
