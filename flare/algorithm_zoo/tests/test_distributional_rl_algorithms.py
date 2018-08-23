from flare.algorithm_zoo.distributional_rl_algorithms import C51
from flare.model_zoo.distributional_rl_models import SimpleModelC51
import numpy as np
import math
import torch
import unittest


class TestC51(unittest.TestCase):
    def test_select_q_distribution(self):
        """
        Test case for selecting a Q distribution with an action.
        """
        alg = C51(model=SimpleModelC51(
            dims=None, num_actions=None, mlp=None, vmax=10, vmin=-10, bins=2),
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
        """
        Test case for backup.
        """
        model = SimpleModelC51(
            dims=None, num_actions=None, mlp=None, vmax=10, vmin=-10, bins=2)

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


if __name__ == "__main__":
    unittest.main()
