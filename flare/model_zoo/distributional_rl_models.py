from flare.framework import common_functions as comf
from simple_models import SimpleModelQ
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleModelC51(SimpleModelQ):
    def __init__(self, dims, num_actions, perception_net, vmax, vmin, bins):
        assert bins > 1
        assert vmax > vmin
        super(SimpleModelC51, self).__init__(dims, num_actions * bins,
                                             perception_net)
        self.num_actions = num_actions
        self.vmax = vmax
        self.vmin = vmin
        self.bins = bins

        self.delta_z = float(self.vmax - self.vmin) / (self.bins - 1)
        atoms = [vmin + i * self.delta_z for i in xrange(self.bins)]
        self.atoms = torch.tensor(atoms)

    def value(self, inputs, states):
        q_distributions = self.value_net(inputs.values()[0])
        q_distributions = q_distributions.view(-1, self.num_actions, self.bins)
        q_distributions = F.softmax(q_distributions, 2)
        q_values = torch.matmul(q_distributions, self.atoms)
        return dict(
            q_value=q_values, q_value_distribution=q_distributions), states


class SimpleModelQRDQN(SimpleModelQ):
    def __init__(self, dims, num_actions, perception_net, N):
        super(SimpleModelQRDQN, self).__init__(dims, num_actions * N,
                                               perception_net)
        self.num_actions = num_actions
        self.N = N

    def value(self, inputs, states):
        q_quantiles = self.value_net(inputs.values()[0])
        q_quantiles = q_quantiles.view(-1, self.num_actions, self.N)
        q_values = q_quantiles.mean(-1)
        return dict(q_value=q_values, q_value_distribution=q_quantiles), states


class SimpleModelIQN(SimpleModelQ):
    def __init__(self,
                 dims,
                 num_actions,
                 perception_net,
                 inner_size,
                 K=32,
                 n=64):
        super(SimpleModelIQN, self).__init__(dims, inner_size, perception_net)
        self.num_actions = num_actions
        self.K = K
        self.inner_size = inner_size
        self.pi_base = torch.tensor([math.pi * i for i in xrange(n)]).view(1,
                                                                           -1)
        self.phi_mlp = nn.Sequential(nn.Linear(n, self.inner_size), nn.ReLU())
        self.f = nn.Linear(self.inner_size, num_actions)

    def get_phi(self, batch_size, N):
        tau = torch.rand(batch_size, N)
        x = tau.view(-1, 1) * self.pi_base
        x = x.cos()
        phi = self.phi_mlp(x)
        phi = phi.view(batch_size, N, -1)
        return phi, tau

    def policy(self, inputs, states):
        values, states = self.value(inputs, states, N=self.K)
        q_value = values["q_value"]
        return dict(action=comf.q_categorical(q_value)), states

    def value(self, inputs, states, N=None):
        if N is None: N = self.N
        psi = self.value_net(inputs.values()[0])
        psi = psi.view(-1, 1, self.inner_size)
        phi, tau = self.get_phi(psi.size()[0], N)
        Z = psi * phi
        Z = Z.view(-1, self.inner_size)
        q_quantiles = self.f(Z)
        q_quantiles = q_quantiles.view(-1, N, self.num_actions)
        q_quantiles = q_quantiles.transpose(1, 2)
        q_values = q_quantiles.mean(-1)
        return dict(
            q_value=q_values, q_value_distribution=q_quantiles,
            tau=tau), states
