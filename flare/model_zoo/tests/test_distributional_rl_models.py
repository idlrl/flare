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


class TestSimpleModelIQN(unittest.TestCase):
    def test_get_phi(self):
        """
        Test case for selecting a Q distribution with an action.
        """

        batch_size = 3
        inner_size = 256
        K = 32
        N = 8
        model = SimpleModelIQN(
            dims=1, num_actions=3, mlp=None, inner_size=inner_size, K=K)
        phi, tau = model.get_phi(batch_size, N)
        self.assertEqual((batch_size, N, inner_size), phi.size())
        self.assertEqual((batch_size, N), tau.size())

    def test_value(self):
        """
        Test case for selecting a Q distribution with an action.
        """
        batch_size = 5
        num_actions = 3
        state_shape = 10
        inner_size = 256
        K = 32
        N = 8
        mlp = nn.Linear(state_shape, inner_size)
        dm = DummyInput(batch_size, state_shape)
        state = None

        model = SimpleModelIQN(
            dims=state_shape,
            num_actions=num_actions,
            mlp=mlp,
            inner_size=inner_size,
            K=K)
        value, a_state = model.value(dm, state, N)
        self.assertEqual(state, a_state)
        self.assertEqual((batch_size, num_actions, N),
                         value["q_value_list"].size())


if __name__ == "__main__":
    unittest.main()
