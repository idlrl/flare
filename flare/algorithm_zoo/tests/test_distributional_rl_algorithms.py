from flare.algorithm_zoo.distributional_rl_algorithms import C51
from flare.model_zoo.distributional_rl_models import SimpleModelC51
from flare.algorithm_zoo.distributional_rl_algorithms import QRDQN
from flare.model_zoo.distributional_rl_models import SimpleModelQRDQN
import numpy as np
import math
import torch
import torch.nn as nn
import unittest


class TestC51(unittest.TestCase):
    def test_select_q_distribution(self):
        inner_size = 256
        num_actions = 3
        state_shape = [1]
        mlp = nn.Sequential(nn.Linear(inner_size, inner_size), nn.ReLU())
        alg = C51(model=SimpleModelC51(
            dims=state_shape,
            num_actions=num_actions,
            perception_net=mlp,
            vmax=10,
            vmin=-10,
            bins=2),
                  exploration_end_steps=500000,
                  update_ref_interval=100)

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
        inner_size = 256
        num_actions = 3
        state_shape = [1]
        mlp = nn.Sequential(nn.Linear(inner_size, inner_size), nn.ReLU())
        alg = C51(model=SimpleModelC51(
            dims=state_shape,
            num_actions=num_actions,
            perception_net=mlp,
            vmax=10,
            vmin=-10,
            bins=3),
                  exploration_end_steps=500000,
                  update_ref_interval=100)

        values = [[[1, 2, 3]] * 2, [[3, 4, 5]] * 2, [[5, 6, 7]] * 2]
        alive = [1, 0, 1]
        next_values = {"q_value_distribution": torch.tensor(values).float()}
        next_alive = {"alive": torch.tensor(alive).float().view(-1, 1)}

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
            bins=2)
        alg = C51(model=model,
                  exploration_end_steps=500000,
                  update_ref_interval=100)

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


class TestQRDQN(unittest.TestCase):
    def test_check_alive(self):
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

        values = [[[1], [2], [3]], [[3], [4], [5]], [[5], [6], [7]]]
        alive = [1, 0, 1]
        next_values = {"q_value_distribution": torch.tensor(values).float()}
        next_alive = {"alive": torch.tensor(alive).float().view(-1, 1)}

        expected = [
            a if b == 1 else [[0], [0], [0]] for a, b in zip(values, alive)
        ]
        expected = np.array(expected)
        actual = alg.check_alive(next_values, next_alive).numpy()
        self.assertEqual(expected.shape, actual.shape)
        for x, y in zip(expected.flatten(), actual.flatten()):
            self.assertAlmostEqual(x, y)

    def test_get_quantile_huber_loss(self):
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

        critic_value = torch.tensor([[-1, 2], [3, 4], [-5, 6]]).float()
        q_distribution = torch.tensor([[9, 8.5], [7, 6], [-5, 6]]).float()
        tau = torch.tensor([[0.3, 0.6]]).float()

        expected = np.array([[4.525], [1.525], [0.0]])
        actual = alg.get_quantile_huber_loss(critic_value, q_distribution,
                                             tau).numpy()
        self.assertEqual(expected.shape, actual.shape)
        for x, y in zip(expected.flatten(), actual.flatten()):
            self.assertAlmostEqual(x, y, places=6)

    def get_quantile_huber_loss(self, critic_value, q_distribution, tau):
        huber_loss = self.loss(q_distribution, critic_value)
        u = critic_value - q_distribution
        delta = (u < 0).float()
        asymmetric = torch.abs(tau - delta)
        quantile_huber_loss = asymmetric * huber_loss
        cost = quantile_huber_loss.mean(1, keepdim=True)
        return cost


if __name__ == "__main__":
    unittest.main()
