from flare.framework.algorithm import Model
from flare.framework import common_functions as comf
import torch
import torch.nn as nn
from torch.distributions import Categorical


class SimpleSRModel(Model):
    def __init__(self,
                 dims,
                 hidden_size,
                 num_actions,
                 autoencoder_levels_n=3,
                 sr_levels_n=3):
        super(SimpleSRModel, self).__init__()
        self.dims = dims
        assert hidden_size % (2**(autoencoder_levels_n - 1)) == 0
        self.num_actions = num_actions

        ### Init the feature encoding network
        encoder_modules = []
        for i in range(autoencoder_levels_n):
            if i == 0:
                encoder_modules.append(nn.Linear(dims, hidden_size))
            else:
                encoder_modules.append(nn.Linear(hidden_size, hidden_size))
            encoder_modules.append(nn.ReLU())

        self.state_embedding_dim = hidden_size
        self.encoder = nn.Sequential(*encoder_modules)

        ### Init the goal vector
        self.goal_vec = nn.Parameter(
            torch.rand(self.state_embedding_dim), requires_grad=True)

        ### Init the successor feature network
        sr_modules = []
        for i in range(sr_levels_n):
            if i == 0:
                sr_modules.append(nn.Linear(dims, hidden_size))
                sr_modules.append(nn.ReLU())
            elif i < sr_levels_n - 1:
                sr_modules.append(nn.Linear(hidden_size, hidden_size))
                sr_modules.append(nn.ReLU())
            else:
                sr_modules.append(
                    nn.Linear(hidden_size, self.state_embedding_dim *
                              self.num_actions))
        self.sr_net = nn.Sequential(*sr_modules)

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def state_embedding(self, inputs):
        return self.encoder(inputs.values()[0])

    def goal(self, inputs):
        """
        For this model, the goal is independent of the inputs
        """
        return self.goal_vec.unsqueeze(0).expand(inputs.values()[0].shape[0],
                                                 -1)

    def sr(self, inputs, states):
        srs = self.sr_net(inputs.values()[0])
        return srs.view(-1, self.num_actions, self.state_embedding_dim), states

    def policy(self, inputs, states):
        """
        We first compute the successor features and then use the goal vector to
        compute the Q values.
        """
        srs, states = self.sr(inputs, states)
        goal = self.goal(inputs).unsqueeze(1).expand(-1, self.num_actions, -1)
        q_value = torch.sum(torch.mul(goal, srs), dim=-1).view(
            -1, self.num_actions)
        return dict(action=comf.q_categorical(q_value)), states
