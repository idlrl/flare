from flare.framework import common_functions as comf
from simple_algorithms import SimpleQ
import torch


class DistributionalAlgorithm(SimpleQ):
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

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):

        if self.update_ref_interval \
                and self.total_batches % self.update_ref_interval == 0:
            ## copy parameters from self.model to self.ref_model
            self.ref_model.load_state_dict(self.model.state_dict())
        self.total_batches += 1

        action = actions["action"]
        reward = rewards["reward"]

        values, states_update = self.model.value(inputs, states)
        q_lists = values["q_value_distribution"]

        with torch.no_grad():
            next_values, next_states_update = self.ref_model.value(next_inputs,
                                                                   next_states)
            next_q_lists = self.check_alive(next_values, next_alive)
            next_expected_q_values = next_values["q_value"]
            _, next_action = next_expected_q_values.max(-1)
            next_action = next_action.unsqueeze(-1)

        assert q_lists.size()[:2] == next_q_lists.size()[:2]

        q_list = self.select_q_distribution(q_lists, action)
        next_q_list = self.select_q_distribution(next_q_lists, next_action)

        cost = self.get_cost(q_list, next_q_list, reward, values, next_values)
        avg_cost = comf.get_avg_cost(cost)
        avg_cost.backward(retain_graph=True)

        return dict(cost=cost), states_update, next_states_update

    def check_alive(self, next_values, next_alive):
        pass

    def get_cost(self, q_list, next_q_list, reward, values, next_values):
        pass


class C51(DistributionalAlgorithm):
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
        self.dead_dist = torch.tensor(dead_dist)
        self.float_vmax = torch.FloatTensor([model.vmax])
        self.float_vmin = torch.FloatTensor([model.vmin])

    def check_alive(self, next_values, next_alive):
        ## if not alive, Q value is deterministically the 0.
        alpha = torch.abs(next_alive["alive"]).view(-1, 1, 1)
        next_q_distributions = next_values["q_value_distribution"] * alpha + \
                               self.dead_dist * (1 - alpha)
        return next_q_distributions

    def get_cost(self, q_list, next_q_list, reward, values, next_values):
        critic_value = self.backup(self.model.atoms, self.float_vmax,
                                   self.float_vmin, self.model.delta_z, reward,
                                   self.discount_factor, next_q_list)
        ## Cross-entropy loss
        cost = -torch.matmul(
            critic_value.unsqueeze(1), q_list.log().unsqueeze(-1)).view(-1, 1)
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
        m = torch.zeros(next_q_distribution.size(), dtype=torch.float)
        m = m.scatter_add_(1, l.long(), ml)
        m = m.scatter_add_(1, u.long(), mu)
        return m


class QuantileAlgorithm(DistributionalAlgorithm):
    def __init__(self,
                 model,
                 gpu_id=-1,
                 discount_factor=0.99,
                 exploration_end_steps=0,
                 exploration_end_rate=0.1,
                 update_ref_interval=100):
        super(QuantileAlgorithm, self).__init__(
            model, gpu_id, discount_factor, exploration_end_steps,
            exploration_end_rate, update_ref_interval)
        self.loss = torch.nn.SmoothL1Loss(reduce=False)

    def check_alive(self, next_values, next_alive):
        next_q_distributions = next_values["q_value_distribution"] * torch.abs(
            next_alive["alive"].view(-1, 1, 1))
        return next_q_distributions

    def get_quantile_huber_loss(self, critic_value, q_distribution, tau):
        huber_loss = self.loss(q_distribution, critic_value)
        u = critic_value - q_distribution
        delta = (u < 0).float()
        asymmetric = torch.abs(tau - delta)
        quantile_huber_loss = asymmetric * huber_loss
        cost = quantile_huber_loss.mean(1, keepdim=True)
        return cost


class QRDQN(QuantileAlgorithm):
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
        N = self.model.N
        self.tau_hat = torch.tensor(
            [(2 * i + 1.) / (2 * N) for i in xrange(N)]).view(1, -1)

    def get_cost(self, q_list, next_q_list, reward, values, next_values):
        critic_value = reward + self.discount_factor * next_q_list
        cost = self.get_quantile_huber_loss(critic_value, q_list, self.tau_hat)
        return cost


class IQN(QuantileAlgorithm):
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
                 N_prime=8):
        super(IQN, self).__init__(model, gpu_id, discount_factor,
                                  exploration_end_steps, exploration_end_rate,
                                  update_ref_interval)
        self.model.N = N
        self.ref_model.N = N_prime
        self.N = N
        self.next_N = N_prime

    def get_cost(self, q_list, next_q_list, reward, values, next_values):
        critic_value = reward + self.discount_factor * next_q_list
        tau = values["tau"].unsqueeze(1)

        batch_size = q_list.size()[0]
        q_distribution = q_list.unsqueeze(1).expand(batch_size, self.next_N,
                                                    self.N)
        critic_value = critic_value.unsqueeze(-1).expand(batch_size,
                                                         self.next_N, self.N)

        cost = self.get_quantile_huber_loss(critic_value, q_distribution, tau)
        cost = cost.squeeze(1).sum(-1, keepdim=True)
        return cost
