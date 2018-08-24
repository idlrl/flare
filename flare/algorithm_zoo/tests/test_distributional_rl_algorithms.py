from flare.algorithm_zoo.distributional_rl_algorithms import C51
from flare.model_zoo.distributional_rl_models import SimpleModelC51
from flare.algorithm_zoo.distributional_rl_algorithms import QRDQN
from flare.model_zoo.distributional_rl_models import SimpleModelQRDQN
from flare.algorithm_zoo.distributional_rl_algorithms import IQN
from flare.model_zoo.distributional_rl_models import SimpleModelIQN
import numpy as np
import math
import torch
import torch.nn as nn
import unittest


class TestC51(unittest.TestCase):
    def initialize(self, bins=2):
        inner_size = 256
        num_actions = 3
        state_shape = [1]
        mlp = nn.Sequential(nn.Linear(inner_size, inner_size), nn.ReLU())
        model = SimpleModelC51(
            dims=state_shape,
            num_actions=num_actions,
            perception_net=mlp,
            vmax=10,
            vmin=-10,
            bins=bins)
        alg = C51(model=model,
                  exploration_end_steps=500000,
                  update_ref_interval=100)
        return model, alg

    def test_select_q_distribution(self):
        model, alg = self.initialize()

        distribution = [[[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]],
                        [[0.4, 0.6], [0.5, 0.5], [0.6, 0.4]]]
        action = [0, 2]

        expected = np.array(
            [d[a] for d, a in zip(distribution, action)]).flatten()
        actual = alg.select_q_distribution(
            torch.tensor(distribution), torch.tensor(action)).numpy().flatten()

        self.assertEqual(len(expected), len(actual))
        for x, y in zip(expected, actual):
            self.assertAlmostEqual(x, y)

    def test_check_alive(self):
        model, alg = self.initialize(3)

        values = [[[1, 2, 3]] * 2, [[3, 4, 5]] * 2, [[5, 6, 7]] * 2]
        alive = [1, 0, 1]
        next_values = torch.tensor(values).float()
        next_alive = torch.tensor(alive).float().view(-1, 1)

        expected = [
            a if b == 1 else [[0, 1, 0]] * 2 for a, b in zip(values, alive)
        ]
        expected = np.array(expected)
        actual = alg.check_alive(next_values, next_alive).numpy()
        self.assertEqual(expected.shape, actual.shape)
        for x, y in zip(expected.flatten(), actual.flatten()):
            self.assertAlmostEqual(x, y)

    def one_backup(self, r, q, discount, model):
        N = len(q)
        m = [0.] * N
        for j in xrange(N):
            Tz = r + discount * model.atoms[j]
            Tz = min(Tz, 10)
            Tz = max(Tz, -10)
            b = (Tz + 10.) / model.delta_z
            l = int(math.floor(b))
            u = int(math.ceil(b))
            m[l] += q[j] * (u - b)
            m[u] += q[j] * (b - l)
        return m

    def test_backup(self):
        model, alg = self.initialize()

        discount = 0.9
        reward = [[1.5], [-0.2], [0.]]
        next_q_distribution = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]]

        expected = np.array([
            self.one_backup(r[0], q, discount, model)
            for r, q in zip(reward, next_q_distribution)
        ]).flatten()

        actual = alg.backup(
            model.atoms,
            torch.FloatTensor([model.vmax]),
            torch.FloatTensor([model.vmin]), model.delta_z,
            torch.tensor(reward), discount,
            torch.tensor(next_q_distribution)).numpy().flatten()

        self.assertEqual(len(expected), len(actual))
        for x, y in zip(expected, actual):
            self.assertAlmostEqual(x, y)

    def test_get_current_values(self):
        model, alg = self.initialize()

        A = "A"
        B = "B"
        alg.model.value = lambda x, y: (x, y)
        A_hat, B_hat = alg.get_current_values(A, B)
        self.assertEqual(A, A_hat)
        self.assertEqual(B, B_hat)

    def test_get_next_values(self):
        model, alg = self.initialize()

        A = "A"
        B = "B"
        C = {"q_value": A}
        alg.ref_model.value = lambda x, y: (x, y)
        C_hat, B_hat, A_hat = alg.get_next_values(C, B)
        self.assertEqual(A, A_hat)
        self.assertEqual(B, B_hat)
        self.assertEqual(C, C_hat)


class TestQRDQN(unittest.TestCase):
    def initialize(self, bins=2):
        inner_size = 256
        num_actions = 3
        state_shape = [1]
        N = 51
        mlp = nn.Sequential(nn.Linear(inner_size, inner_size), nn.ReLU())
        alg = QRDQN(
            model=SimpleModelQRDQN(
                dims=state_shape,
                num_actions=num_actions,
                perception_net=mlp,
                N=N),
            exploration_end_steps=500000,
            update_ref_interval=100)
        return alg

    def test_check_alive(self):
        alg = self.initialize()

        values = [[[1], [2], [3]], [[3], [4], [5]], [[5], [6], [7]]]
        alive = [1, 0, 1]
        next_values = torch.tensor(values).float()
        next_alive = torch.tensor(alive).float().view(-1, 1)

        expected = [
            a if b == 1 else [[0], [0], [0]] for a, b in zip(values, alive)
        ]
        expected = np.array(expected)
        actual = alg.check_alive(next_values, next_alive).numpy()
        self.assertEqual(expected.shape, actual.shape)
        for x, y in zip(expected.flatten(), actual.flatten()):
            self.assertAlmostEqual(x, y)

    def huber_loss(self, u, k=1):
        if abs(u) <= k:
            return 0.5 * u * u
        else:
            return k * (abs(u) - 0.5 * k)

    def quantile_huber_loss(self, u, tau, k=1):
        if u < 0:
            delta = 1
        else:
            delta = 1
        return abs(tau - delta) * self.huber_loss(u, k)

    def expection_quantile_huber_loss(self, theta, Ttheta, tau, k=1):
        r1 = 0
        for theta_i, tau_i in zip(theta, tau):
            r2 = 0
            for Ttheta_j in Ttheta:
                r2 += self.quantile_huber_loss(Ttheta_j - theta_i, tau_i, k)
            r1 += r2 / len(Ttheta)
        return r1

    def batch_expection_quantile_huber_loss(self,
                                            q_distribution,
                                            critic_value,
                                            tau,
                                            k=1):
        expected = []
        for theta, Ttheta, t in zip(q_distribution, critic_value, tau):
            expected.append(
                self.expection_quantile_huber_loss(theta, Ttheta, t, k))
        return expected

    def test_get_quantile_huber_loss(self):
        alg = self.initialize()

        critic_value = [[-1., 2.], [3., 4.], [-5., -5.]]
        q_distribution = [[9., 8.5], [7., 6.], [-5., -5.]]
        tau = [[0.3, 0.6], [0.4, 0.8], [0.6, 0.1]]
        expected = self.batch_expection_quantile_huber_loss(
            q_distribution, critic_value, tau, k=1)
        expected = np.array(expected)

        critic_value = torch.tensor(critic_value)
        q_distribution = torch.tensor(q_distribution)
        tau = torch.tensor(tau)
        actual = alg.get_quantile_huber_loss(critic_value, q_distribution,
                                             tau).view(-1).numpy()

        self.assertEqual(expected.shape, actual.shape)
        for x, y in zip(expected.flatten(), actual.flatten()):
            self.assertAlmostEqual(x, y, places=6)


class TestIQN(unittest.TestCase):
    def initialize(self):
        inner_size = 256
        num_actions = 3
        state_shape = [1]
        mlp = nn.Sequential(nn.Linear(inner_size, inner_size), nn.ReLU())
        model = SimpleModelIQN(
            dims=state_shape,
            num_actions=num_actions,
            perception_net=mlp,
            inner_size=inner_size)
        alg = IQN(model=model,
                  exploration_end_steps=500000,
                  update_ref_interval=100)
        return alg

    def test_get_current_values(self):
        alg = self.initialize()

        A = "A"
        B = "B"
        N = 10
        alg.model.value = lambda x, y, z: (x, y, z)
        alg.N = N
        A_hat, B_hat, N_hat = alg.get_current_values(A, B)
        self.assertEqual(A, A_hat)
        self.assertEqual(B, B_hat)
        self.assertEqual(N, N_hat)

    def tdest_get_next_values(self):
        alg = self.initialize()

        next_values = "A"
        next_value = "B"
        next_states_update = "C"
        a_list = [next_values, {"q_value": next_value}]
        alg.ref_model.value = lambda x, y, z: (x, y, z)

        A_hat, B_hat, C_hat = alg.get_next_values(a_list, next_states_update)
        self.assertEqual(next_values, A_hat)
        self.assertEqual(next_states_update, B_hat)
        self.assertEqual(next_value, C_hat)


if __name__ == "__main__":
    unittest.main()
