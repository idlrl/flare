from flare.model_zoo.distributional_rl_models import SimpleModelC51
from flare.model_zoo.distributional_rl_models import SimpleModelQRDQN
from flare.model_zoo.distributional_rl_models import SimpleModelIQN
import torch
import torch.nn as nn
import unittest


class DummyInput():
    def __init__(self, batch_size, state_shape):
        self.batch_size = batch_size
        self.state_shape = state_shape

    def values(self):
        return [torch.randn(self.batch_size, self.state_shape)]


class TestSimpleModelC51(unittest.TestCase):
    def test_value(self):
        batch_size = 5
        num_actions = 3
        state_shape = [10]
        bins = 51
        state = None
        mlp = nn.Sequential(nn.Linear(state_shape[0], 256), nn.ReLU())
        dm = DummyInput(batch_size, state_shape[0])

        model = SimpleModelC51(
            dims=state_shape,
            num_actions=num_actions,
            perception_net=mlp,
            vmax=10,
            vmin=-10,
            bins=bins)
        value, a_state = model.value(dm, state)
        q_value_distribution = value["q_value_distribution"]
        q_value = value["q_value"]
        self.assertEqual(state, a_state)
        self.assertEqual((batch_size, num_actions, bins),
                         q_value_distribution.size())
        self.assertEqual((batch_size, num_actions), q_value.size())
        for i in xrange(batch_size):
            for j in xrange(num_actions):
                self.assertAlmostEqual(
                    1.,
                    sum(q_value_distribution[i][j].data.tolist()),
                    places=6)


class TestSimpleSimpleModelQRDQN(unittest.TestCase):
    def test_value(self):
        batch_size = 5
        num_actions = 3
        state_shape = [10]
        N = 51
        state = None
        mlp = nn.Sequential(nn.Linear(state_shape[0], 256), nn.ReLU())
        dm = DummyInput(batch_size, state_shape[0])

        model = SimpleModelQRDQN(
            dims=state_shape, num_actions=num_actions, perception_net=mlp, N=N)
        value, a_state = model.value(dm, state)
        q_value_distribution = value["q_value_distribution"]
        q_value = value["q_value"]
        self.assertEqual(state, a_state)
        self.assertEqual((batch_size, num_actions, N),
                         q_value_distribution.size())
        self.assertEqual((batch_size, num_actions), q_value.size())
        for i in xrange(batch_size):
            for j in xrange(num_actions):
                self.assertAlmostEqual(
                    q_value[i][j].data.tolist(),
                    sum(q_value_distribution[i][j].data.tolist()) / N)


class TestSimpleModelIQN(unittest.TestCase):
    def test_value(self):
        batch_size = 5
        num_actions = 3
        state_shape = [10]
        inner_size = 256
        K = 32
        N = 8
        state = None
        mlp = nn.Sequential(nn.Linear(state_shape[0], inner_size), nn.ReLU())
        dm = DummyInput(batch_size, state_shape[0])

        model = SimpleModelIQN(
            dims=state_shape,
            num_actions=num_actions,
            perception_net=mlp,
            inner_size=inner_size,
            default_samples=K)
        value, a_state = model.value(dm, state, N)
        self.assertEqual(state, a_state)
        self.assertEqual((batch_size, num_actions, N),
                         value["q_value_distribution"].size())

    def test_values(self):
        batch_size = 5
        num_actions = 3
        state_shape = [10]
        inner_size = 256
        K = 32
        N1 = 8
        N2 = 5
        state = None
        mlp = nn.Sequential(nn.Linear(state_shape[0], inner_size), nn.ReLU())
        dm = DummyInput(batch_size, state_shape[0])

        model = SimpleModelIQN(
            dims=state_shape,
            num_actions=num_actions,
            perception_net=mlp,
            inner_size=inner_size,
            default_samples=K)
        values, a_state = model.values(dm, state, [N1, N2])
        self.assertEqual(state, a_state)
        self.assertEqual((batch_size, num_actions, N1),
                         values[0]["q_value_distribution"].size())
        self.assertEqual((batch_size, num_actions, N2),
                         values[1]["q_value_distribution"].size())

    def test_get_phi(self):
        batch_size = 3
        K = 32
        N = 8
        state_shape = [1]
        inner_size = 256
        mlp = nn.Sequential(nn.Linear(inner_size, inner_size), nn.ReLU())
        model = SimpleModelIQN(
            dims=state_shape,
            num_actions=3,
            perception_net=mlp,
            inner_size=inner_size,
            default_samples=K)
        phi, tau = model.get_phi(batch_size, N)
        self.assertEqual((batch_size, N, inner_size), phi.size())
        self.assertEqual((batch_size, N), tau.size())


if __name__ == "__main__":
    unittest.main()
