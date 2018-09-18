from flare.framework.algorithm import Model, Algorithm
from flare.framework import common_functions as comf
from flare.model_zoo.simple_models import SimpleModelDeterministic
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import unittest


class TestAlgorithm(Algorithm):
    def __init__(self, model):
        super(TestAlgorithm, self).__init__(model, gpu_id=-1)

    def predict(self, inputs, states):
        return self._rl_predict(self.model, inputs, states)


class TestAlgorithmParas(unittest.TestCase):
    def test_copy_paras(self):
        """
        Test case for copying parameters
        """

        alg1 = TestAlgorithm(model=SimpleModelDeterministic(
            dims=[10], perception_net=nn.Linear(10, 10)))
        alg2 = TestAlgorithm(model=SimpleModelDeterministic(
            dims=[10], perception_net=nn.Linear(10, 10)))

        batch_size = 10
        sensor = torch.tensor(
            np.random.uniform(0, 1, [batch_size] + alg1.model.dims).astype(
                "float32"))

        with torch.no_grad():
            outputs1, _ = alg1.predict(
                inputs=dict(sensor=sensor), states=dict())
            outputs2, _ = alg2.predict(
                inputs=dict(sensor=sensor), states=dict())

        self.assertNotEqual(
            np.sum(outputs1.values()[0].numpy().flatten()),
            np.sum(outputs2.values()[0].numpy().flatten()))

        ## do the copying
        alg1.model.load_state_dict(alg2.model.state_dict())

        with torch.no_grad():
            outputs1, _ = alg1.predict(
                inputs=dict(sensor=sensor), states=dict())
            outputs2, _ = alg2.predict(
                inputs=dict(sensor=sensor), states=dict())

        self.assertEqual(
            np.sum(outputs1.values()[0].numpy().flatten()),
            np.sum(outputs2.values()[0].numpy().flatten()))

    def test_share_paras(self):
        """
        Test case for sharing parameters
        """
        layer = nn.Linear(10, 10)
        alg1 = TestAlgorithm(model=SimpleModelDeterministic(
            dims=[10], perception_net=layer))
        alg2 = TestAlgorithm(model=SimpleModelDeterministic(
            dims=[10], perception_net=layer))
        alg3 = TestAlgorithm(model=SimpleModelDeterministic(
            dims=[10], perception_net=nn.Sequential(layer)))
        alg4 = deepcopy(alg1)

        batch_size = 10
        sensor = torch.tensor(
            np.random.uniform(0, 1, [batch_size] + alg1.model.dims).astype(
                "float32"))

        with torch.no_grad():
            outputs1, _ = alg1.predict(
                inputs=dict(sensor=sensor), states=dict())
            outputs2, _ = alg2.predict(
                inputs=dict(sensor=sensor), states=dict())
            outputs3, _ = alg3.predict(
                inputs=dict(sensor=sensor), states=dict())
            outputs4, _ = alg4.predict(
                inputs=dict(sensor=sensor), states=dict())

        self.assertEqual(
            np.sum(outputs1.values()[0].numpy().flatten()),
            np.sum(outputs2.values()[0].numpy().flatten()))
        self.assertEqual(
            np.sum(outputs2.values()[0].numpy().flatten()),
            np.sum(outputs3.values()[0].numpy().flatten()))
        self.assertEqual(
            np.sum(outputs3.values()[0].numpy().flatten()),
            np.sum(outputs4.values()[0].numpy().flatten()))


if __name__ == "__main__":
    unittest.main()
