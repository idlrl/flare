from simple_models import SimpleModelQ
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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

    def value(self, inputs, states):
        q_distributions = self.mlp(inputs.values()[0])
        q_distributions = q_distributions.view(-1, self.num_actions, self.bins)
        q_distributions = F.softmax(q_distributions, 2)
        q_values = torch.matmul(q_distributions, self.atoms)
        return dict(q_value=q_values, q_value_list=q_distributions), states



class SimpleModelQRDQN(SimpleModelQ):
    def __init__(self, dims, num_actions, mlp, N):
        super(SimpleModelQRDQN, self).__init__(dims, num_actions, mlp)
        self.N = N

    def value(self, inputs, states):
        q_quantiles = self.mlp(inputs.values()[0])
        q_quantiles = q_quantiles.view(-1, self.num_actions, self.N)
        q_values = q_quantiles.mean(-1)
        return dict(q_value=q_values, q_value_list=q_quantiles), states


class SimpleModelIQN(SimpleModelQ):
    def __init__(self, dims, num_actions, mlp, inner_size, K=32, n=64):
        super(SimpleModelIQN, self).__init__(dims, num_actions, mlp)
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
        q_quantiles = self.f(Z)
        q_quantiles = q_quantiles.view(-1, N, self.num_actions)
        q_quantiles = q_quantiles.transpose(1, 2)
        q_values = q_quantiles.mean(-1)
        return dict(q_value=q_values, q_value_list=q_quantiles, tau=tau), states
