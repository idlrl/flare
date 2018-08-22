from flare.framework.algorithm import Model
from flare.framework.distributions import Deterministic
from flare.framework import common_functions as comf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal
import math


class SimpleModelDeterministic(Model):
    def __init__(self, dims, mlp):
        super(SimpleModelDeterministic, self).__init__()
        self.dims = dims
        self.mlp = mlp

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("continuous_action", dict(shape=[self.dims]))]

    def policy(self, inputs, states):
        hidden = self.mlp(inputs.values()[0])
        return dict(continuous_action=Deterministic(hidden)), states


class SimpleModelAC(Model):
    def __init__(self, dims, num_actions, mlp):
        super(SimpleModelAC, self).__init__()
        self.dims = dims
        hidden_size = list(mlp.modules())[-2].out_features
        self.policy_net = nn.Sequential(
            mlp, nn.Linear(hidden_size, num_actions), nn.Softmax(dim=1))
        self.value_net = nn.Sequential(mlp, nn.Linear(hidden_size, 1))

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def policy(self, inputs, states):
        dist = Categorical(probs=self.policy_net(inputs.values()[0]))
        return dict(action=dist), states

    def value(self, inputs, states):
        return dict(v_value=self.value_net(inputs.values()[0])), states


class SimpleModelQ(Model):
    def __init__(self, dims, num_actions, mlp):
        super(SimpleModelQ, self).__init__()
        self.dims = dims
        self.num_actions = num_actions
        self.mlp = mlp

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def policy(self, inputs, states):
        values, states = self.value(inputs, states)
        q_value = values["q_value"]
        return dict(action=comf.q_categorical(q_value)), states

    def value(self, inputs, states):
        return dict(q_value=self.mlp(inputs.values()[0])), states


class SimpleModelC51(SimpleModelQ):
    def __init__(self, dims, num_actions, mlp, vmax, vmin, bins):
        super(SimpleModelC51, self).__init__(dims, num_actions, mlp)
        assert bins > 1
        assert vmax > vmin
        self.vmax = vmax
        self.vmin = vmin
        self.bins = bins

        self.delta_z = float(self.vmax - self.vmin) / (self.bins - 1)
        atoms = [vmin + i * self.delta_z for i in xrange(self.bins)]
        self.atoms = torch.tensor(atoms)

    def policy(self, inputs, states):
        values, states = self.value(inputs, states)
        expected_q_values = self.get_expected_q_values(values["q_value"])
        return dict(action=comf.q_categorical(expected_q_values)), states

    def value(self, inputs, states):
        q_distributions = self.mlp(inputs.values()[0])
        q_distributions = q_distributions.view(-1, self.num_actions, self.bins)
        q_distributions = F.softmax(q_distributions, 2)
        return dict(q_value=q_distributions), states

    def get_expected_q_values(self, q_distribution):
        return torch.matmul(q_distribution, self.atoms)


class SimpleModelQRDQN(SimpleModelC51):
    def __init__(self, dims, num_actions, mlp, N):
        super(SimpleModelQRDQN, self).__init__(dims, num_actions, mlp, 10, -10, N)
        self.N = N

    def get_expected_q_values(self, q_distribution):
        return q_distribution.mean(-1)


class SimpleModelIQN(SimpleModelQRDQN):
    def __init__(self, dims, num_actions, mlp, inner_size, K=32, n=64):
        super(SimpleModelIQN, self).__init__(dims, num_actions, mlp, K)
        self.K = K
        self.inner_size = inner_size
        self.pi_base = torch.tensor([math.pi * i for i in xrange(n)]).view(1, -1)
        self.phi_mlp = nn.Sequential(
            nn.Linear(n, self.inner_size),
            nn.ReLU())
        self.f = nn.Linear(self.inner_size, num_actions)

    def get_phi(self, batch_size, N):
        tau = torch.rand(batch_size, N)
        x = tau.view(-1, 1) * self.pi_base
        x = x.cos()
        phi = self.phi_mlp(x)
        phi = phi.view(batch_size, N, -1)
        return phi, tau

    def value(self, inputs, states, N=None):
        if N is None:
            N = self.K
        psi = self.mlp(inputs.values()[0])
        psi = psi.view(-1, 1, self.inner_size)
        phi, tau = self.get_phi(psi.size()[0], N)
        Z = psi * phi
        Z = Z.view(-1, self.inner_size)
        q_values = self.f(Z)
        q_values = q_values.view(-1, N, self.num_actions)
        q_values = q_values.transpose(1, 2)
        return dict(q_value=q_values, tau=tau), states


class SimpleRNNModelAC(Model):
    def __init__(self, dims, num_actions, mlp):
        super(SimpleRNNModelAC, self).__init__()
        self.dims = dims
        self.num_actions = num_actions
        self.hidden_size = list(mlp.children())[-2].out_features
        self.hidden_layers = mlp
        self.recurrent = nn.RNNCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            nonlinearity="relu")
        self.policy_layers = nn.Sequential(
            nn.Linear(self.hidden_size, num_actions), nn.Softmax(dim=1))
        self.value_layer = nn.Linear(self.hidden_size, 1)

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def get_state_specs(self):
        return [("state", dict(shape=[self.hidden_size]))]

    def policy(self, inputs, states):
        hidden = self.hidden_layers(inputs.values()[0])
        next_state = self.recurrent(hidden, states.values()[0])
        dist = Categorical(probs=self.policy_layers(next_state))
        return dict(action=dist), dict(state=next_state)

    def value(self, inputs, states):
        hidden = self.hidden_layers(inputs.values()[0])
        next_state = self.recurrent(hidden, states.values()[0])
        return dict(v_value=self.value_layer(next_state)), dict(
            state=next_state)


class SimpleRNNModelQ(Model):
    def __init__(self, dims, num_actions, mlp):
        super(SimpleRNNModelQ, self).__init__()
        self.dims = dims
        self.num_actions = num_actions
        self.hidden_layers = mlp
        self.hidden_size = list(mlp.children())[-2].out_features
        self.value_layer = nn.Linear(self.hidden_size, num_actions)
        self.recurrent = nn.RNNCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            nonlinearity="relu")

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def get_state_specs(self):
        return [("state", dict(shape=[self.hidden_size]))]

    def policy(self, inputs, states):
        values, next_states = self.value(inputs, states)
        q_value = values["q_value"]
        return dict(action=comf.q_categorical(q_value)), next_states

    def value(self, inputs, states):
        hidden = self.hidden_layers(inputs.values()[0])
        next_state = self.recurrent(hidden, states.values()[0])
        return dict(q_value=self.value_layer(next_state)), dict(
            state=next_state)


class GaussianPolicyModel(Model):
    def __init__(self, dims, action_dims, mlp, std=0.01):
        super(GaussianPolicyModel, self).__init__()
        self.dims = dims
        self.action_dims = action_dims
        self.std = std
        hidden_size = list(mlp.modules())[-2].out_features
        self.policy_net = nn.Sequential(mlp,
                                        nn.Linear(hidden_size, action_dims))
        self.value_net = nn.Sequential(mlp, nn.Linear(hidden_size, 1))

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[self.action_dims]))]

    def policy(self, inputs, states):
        dist = MultivariateNormal(
            loc=self.policy_net(inputs.values()[0]),
            covariance_matrix=torch.eye(self.action_dims) * self.std)
        return dict(action=dist), states

    def value(self, inputs, states):
        return dict(v_value=self.value_net(inputs.values()[0])), states
