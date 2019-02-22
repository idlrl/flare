from flare.model_zoo.simple_models import SimpleModelQ
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class C51Model(SimpleModelQ):
    def __init__(self,
                 dims,
                 num_actions,
                 perception_net,
                 vmax=10,
                 vmin=-10,
                 bins=51):
        assert bins > 1
        assert vmax > vmin
        super(C51Model, self).__init__(dims, num_actions * bins,
                                       perception_net)
        self.num_actions = num_actions
        self.vmax = vmax
        self.vmin = vmin
        self.bins = bins

        self.delta_z = float(self.vmax - self.vmin) / (self.bins - 1)
        atoms = [vmin + i * self.delta_z for i in range(self.bins)]
        atoms = torch.tensor(atoms, requires_grad=False)
        ## atoms are constent, so register as buffer.
        self.register_buffer("atoms", atoms)

    def value(self, inputs, states):
        q_distributions = self.value_net(inputs["sensor"])
        q_distributions = q_distributions.view(-1, self.num_actions, self.bins)
        q_distributions = F.softmax(q_distributions, 2)
        q_values = torch.matmul(q_distributions, self.atoms)
        return dict(
            q_value=q_values, q_value_distribution=q_distributions), states


class QRDQNModel(SimpleModelQ):
    def __init__(self, dims, num_actions, perception_net, N=32):
        assert N > 0
        super(QRDQNModel, self).__init__(dims, num_actions * N, perception_net)
        self.num_actions = num_actions
        tau_hat = torch.tensor(
            [(2 * i + 1.) / (2 * N) for i in range(N)],
            requires_grad=False).view(1, -1)
        self.register_buffer("tau_hat", tau_hat)
        self.N = N

    def value(self, inputs, states):
        q_quantiles = self.value_net(inputs["sensor"])
        q_quantiles = q_quantiles.view(-1, self.num_actions, self.N)
        q_values = q_quantiles.mean(-1)
        return dict(
            q_value=q_values,
            q_value_distribution=q_quantiles,
            tau=self.tau_hat), states


class IQNModel(SimpleModelQ):
    def __init__(self,
                 dims,
                 num_actions,
                 perception_net,
                 inner_size=256,
                 n=64,
                 default_samples=32):
        assert inner_size > 0
        assert n > 0
        assert default_samples > 0
        super(IQNModel, self).__init__(dims, inner_size, perception_net)
        self.num_actions = num_actions
        self.inner_size = inner_size
        pi_base = torch.tensor(
            [math.pi * i for i in range(n)], requires_grad=False).view(1, -1)
        self.register_buffer("pi_base", pi_base)
        phi_mlp = nn.Sequential(nn.Linear(n, self.inner_size), nn.ReLU())
        self.add_module("phi_mlp", phi_mlp)
        self.add_module("f", nn.Linear(self.inner_size, num_actions))
        self.default_samples = default_samples

    def value(self, inputs, states, N=None):
        if N is None: N = self.default_samples
        values, states = self.values(inputs, states, [N])
        return values[0], states

    def values(self, inputs, states, nums):
        """
        Get a list of values with given sample numbers.
        :param inputs: same as value()
        :param states: same as value()
        :param nums: List(int). List of sample numbers.
        :return: List(Dict), states. Dict is the same as return in value()
        """
        assert len(nums) > 0
        psi = self.value_net(inputs.values()[0])
        psi = psi.view(-1, 1, self.inner_size)
        values = []
        for N in nums:
            phi, tau = self.get_phi(psi.size()[0], N)
            Z = psi * phi
            Z = Z.view(-1, self.inner_size)
            q_quantiles = self.f(Z)
            q_quantiles = q_quantiles.view(-1, N, self.num_actions)
            q_quantiles = q_quantiles.transpose(1, 2)
            q_values = q_quantiles.mean(-1)
            values.append(
                dict(
                    q_value=q_values,
                    q_value_distribution=q_quantiles,
                    tau=tau))
        return values, states

    def get_phi(self, batch_size, N):
        """
        Get random Phi values with given shape
        :param batch_size: int. Batch size.
        :param N: int. Number of Phi values for each sample in batch.
            sample in a batch.
        :return: Tensor (batch_size x N). Generated random Phi values.
        """
        tau = torch.Tensor(batch_size, N)
        if self.pi_base.is_cuda:
            tau = tau.to(self.pi_base.get_device())
        tau.normal_()
        x = tau.view(-1, 1) * self.pi_base
        x = x.cos()
        phi = self.phi_mlp(x)
        phi = phi.view(batch_size, N, -1)
        return phi, tau
