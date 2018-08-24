from flare.framework import common_functions as comf
from simple_algorithms import SimpleQ
import torch


class DistributionalAlgorithm(SimpleQ):
    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):

        if self.update_ref_interval \
                and self.total_batches % self.update_ref_interval == 0:
            ## copy parameters from self.model to self.ref_model
            self.ref_model.load_state_dict(self.model.state_dict())
        self.total_batches += 1

        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.get_current_values(inputs, states)
        q_distributions = values["q_value_distribution"]
        q_distribution = self.select_q_distribution(q_distributions, action)

        with torch.no_grad():
            next_values, next_states_update, next_value = self.get_next_values(
                next_inputs, next_states)

            filter = next_alive["alive"]
            next_expected_q_values = next_value * torch.abs(filter)
            _, next_action = next_expected_q_values.max(-1)
            next_q_distributions = self.check_alive(
                next_values["q_value_distribution"], filter)
            assert q_distributions.size()[0] == next_q_distributions.size()[0]
            assert q_distributions.size()[1] == next_q_distributions.size()[1]

            next_action = next_action.unsqueeze(-1)
            next_q_distribution = self.select_q_distribution(
                next_q_distributions, next_action)
            assert q_distribution.size()[0] == next_q_distribution.size()[0]

        cost = self.get_cost(q_distribution, next_q_distribution, reward,
                             values, next_values)
        avg_cost = comf.get_avg_cost(cost)
        avg_cost.backward(retain_graph=True)

        return dict(cost=cost), states_update, next_states_update

    def select_q_distribution(self, q_distributions, action):
        """
        Select a Q value distribution according to a given action.
        :param q_distributions: Tensor (batch_size x num_actions x num_atoms).
            Q value distribution for each action. The histogram has a support
            of length num_atoms.
        :param action: Tensor (batch_size x 1). Index of actions for each
            sample in a batch.
        :return: Tensor (batch_size x num_atoms). Q value distribution for an
            action.
        """
        one_hot_action = comf.one_hot(
            action.squeeze(-1), q_distributions.size()[1])
        one_hot_action = one_hot_action.unsqueeze(1)
        q_distribution = torch.matmul(one_hot_action, q_distributions)
        return q_distribution.squeeze(1)

    def get_current_values(self, inputs, states):
        """
        Get current values from network
        :param inputs: Tensor. Input
        :param states: Tensor. States
        :return: Dict, Tensor. Values and states
        """
        return self.model.value(inputs, states)

    def get_next_values(self, next_inputs, next_states):
        """
        Get current values from network
        :param inputs: Tensor. Input
        :param states: Tensor. States
        :return: Dict, Tensor, Tensor. Values states, and next q values.
        """
        next_values, next_states_update = self.ref_model.value(next_inputs,
                                                               next_states)
        next_value = next_values["q_value"]
        return next_values, next_states_update, next_value

    def check_alive(self, next_values, next_alive):
        """
        Check if an agent is alive and set reward to 0 if not.
        :param next_values: Tensor (batch_size x num_actions x num_atoms).
        :param next_alive: Tensor (batch_size x 1).
        :return: Tensor (batch_size x num_actions x num_atoms).
            Updated Q value distribution.
        """
        pass

    def get_cost(self, q_distribution, next_q_distribution, reward, values,
                 next_values):
        """
        Get cost from current distribution, reward and next values.
        :param q_distribution: Tensor (batch_size x num_atoms).
            Q value distribution.
        :param next_q_distribution: Tensor (batch_size x num_atoms).
            Next Q value distribution.
        :param reward: Tensor (batch_size x 1). Rewards.
        :param next_values: Dict.
        :param next_alive: Dict.
        :return: Tensor (batch_size x 1). Cost.
        """
        pass


class C51(DistributionalAlgorithm):
    """
    Categorical Algorithm (C51) based on SimpleQ.
    Refer https://arxiv.org/pdf/1707.06887.pdf for more details.
    "A distributional perspective on reinforcement learning"

    self.model should have members defined in SimpleModelC51 class.
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 exploration_end_steps=0,
                 exploration_end_rate=0.1,
                 update_ref_interval=100):
        super(C51, self).__init__(model, gpu_id, discount_factor,
                                  exploration_end_steps, exploration_end_rate,
                                  update_ref_interval)
        dead_dist = [0.] * self.model.bins
        dead_dist[len(dead_dist) / 2] = 1.
        self.dead_dist = torch.tensor(
            dead_dist, requires_grad=False,
            device=self.device).view(-1, 1, len(dead_dist))
        self.float_vmax = torch.tensor(
            [model.vmax], requires_grad=False, device=self.device).float()
        self.float_vmin = torch.tensor(
            [model.vmin], requires_grad=False, device=self.device).float()

    def check_alive(self, next_values, next_alive):
        ## if not alive, Q value is deterministically the 0.
        alpha = torch.abs(next_alive).view(-1, 1, 1)
        next_q_distributions = next_values * alpha + self.dead_dist * (1 -
                                                                       alpha)
        return next_q_distributions

    def get_cost(self, q_distribution, next_q_distribution, reward, values,
                 next_values):
        critic_value = self.backup(self.model.atoms, self.float_vmax,
                                   self.float_vmin, self.model.delta_z, reward,
                                   self.discount_factor, next_q_distribution)
        ## Cross-entropy loss
        cost = -torch.matmul(
            critic_value.unsqueeze(1),
            q_distribution.log().unsqueeze(-1)).view(-1, 1)
        return cost

    def backup(self, z, vmax, vmin, delta_z, reward, discount,
               next_q_distribution):
        """
        Backup sampled reward and reference q value distribution to current q
        value ditribution.
        :param z: Tensor (num_atoms). Atoms.
        :param vmax: FloatTensor (1). Maximum value for the distribution.
        :param vmin: FloatTensor (1). Minumum value for the distribution.
        :param delta_z: float. Size of bin for the distribution.
        :param reward: Tensor (batch_size, 1). Reward function.
        :param discount: float. Discount factor.
        :param next_q_distribution: Tensor (batch_size x num_atoms). Q value
            distribution.
        :return: Tensor (batch_size x num_atoms). Q value distribution.
        """
        ## Compute the projection of Tz onto the support z
        Tz = reward + discount * z
        Tz = torch.min(Tz, vmax)
        Tz = torch.max(Tz, vmin)
        b = (Tz - vmin) / delta_z
        l = b.floor()
        u = b.ceil()

        ## Distribute probability of Tz
        wl = u - b
        wu = b - l
        ml = wl * next_q_distribution
        mu = wu * next_q_distribution
        m = torch.zeros(
            next_q_distribution.size(), dtype=torch.float, device=self.device)
        m = m.scatter_add_(1, l.long(), ml)
        m = m.scatter_add_(1, u.long(), mu)
        return m


class QRDQN(DistributionalAlgorithm):
    """
    Quantile Regression DQN (QR-DQN) based on C51.
    Refer to https://arxiv.org/pdf/1710.10044.pdf for more details.
    "Distributional Reinforcement Learning with Quantile Regression"

    self.model should have members defined in SimpleQRDQN class.
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 exploration_end_steps=0,
                 exploration_end_rate=0.1,
                 update_ref_interval=100):
        super(QRDQN, self).__init__(model, gpu_id, discount_factor,
                                    exploration_end_steps,
                                    exploration_end_rate, update_ref_interval)
        self.loss = torch.nn.SmoothL1Loss(reduction='none')

    def check_alive(self, next_values, next_alive):
        next_q_distributions = next_values * torch.abs(
            next_alive.view(-1, 1, 1))
        return next_q_distributions

    def get_cost(self, q_distribution, next_q_distribution, reward, values,
                 next_values):
        tau = values["tau"]
        critic_value = reward + self.discount_factor * next_q_distribution
        cost = self.get_quantile_huber_loss(critic_value, q_distribution, tau)
        return cost

    def get_quantile_huber_loss(self, critic_value, q_distribution, tau):
        """
        Quantile Huber loss mentioned in https://arxiv.org/pdf/1710.10044.pdf
        :param critic_value: Tensor (batch x num_quantiles_a). Target values.
        :param q_distribution: Tensor (batch x num_quantiles_b). Actual values.
        :param tau: Tensor (batch x num_quantiles_a). CDF values. First
            dimention can be broadcaseted.
        :return: Tensor (batch x 1). loss
        """
        ## reshape input tensors
        batch_size, N = q_distribution.size()
        next_batch_size, next_N = critic_value.size()
        assert batch_size == next_batch_size
        assert N == tau.size()[1]
        q_distribution = q_distribution.unsqueeze(1).expand(batch_size, next_N,
                                                            N)
        critic_value = critic_value.unsqueeze(-1).expand(batch_size, next_N, N)
        tau = tau.view(-1, 1, N)

        ## compute loss
        huber_loss = self.loss(q_distribution, critic_value)
        u = critic_value - q_distribution
        delta = u.lt(0).float()
        asymmetric = torch.abs(tau - delta)
        quantile_huber_loss = asymmetric * huber_loss
        cost = quantile_huber_loss.mean(1)
        cost = cost.sum(-1, keepdim=True)
        return cost


class IQN(QRDQN):
    """
    Implicit Quantile Networks (IQN) based on QR-DQN.
    Refer to https://arxiv.org/pdf/1806.06923.pdf for more details.
    "Implicit Quantile Networks for Distributional Reinforcement Learning"

    self.model should have members defined in SimpleIQN class.
    """

    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 exploration_end_steps=0,
                 exploration_end_rate=0.1,
                 update_ref_interval=100,
                 N=8,
                 N_prime=8,
                 K=32):
        super(IQN, self).__init__(model, gpu_id, discount_factor,
                                  exploration_end_steps, exploration_end_rate,
                                  update_ref_interval)
        self.N = N
        self.N_prime = N_prime
        self.K = K

    def get_current_values(self, inputs, states):
        return self.model.value(inputs, states, self.N)

    def get_next_values(self, next_inputs, next_states):
        next_value_list, next_states_update = self.ref_model.values(
            next_inputs, next_states, [self.N_prime, self.K])
        next_values = next_value_list[0]
        next_value = next_value_list[1]["q_value"]
        return next_values, next_states_update, next_value
