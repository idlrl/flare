from flare.model_zoo.sac_models import SACModel
import torch
import torch.nn as nn
import unittest


class DummyInput():
    def __init__(self, batch_size, state_shape):
        self.batch_size = batch_size
        self.state_shape = state_shape

    def values(self):
        return [torch.randn(self.batch_size, self.state_shape)]


class TestSACModelModel(unittest.TestCase):
    def test_value(self):
        batch_size = 5
        num_actions = 3
        state_shape = [10]
        state = None
        mlp = nn.Sequential(nn.Linear(state_shape[0], 256), nn.ReLU())
        dm = DummyInput(batch_size, state_shape[0])

        model = SACModel(
            dims=state_shape, num_actions=num_actions, perception_net=mlp)
        value, a_state = model.value(dm, state)
        assert "q_value" in value
        q_value = value["q_value"]
        self.assertEqual((batch_size, num_actions), q_value.size())


if __name__ == "__main__":
    unittest.main()
