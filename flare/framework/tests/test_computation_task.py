from flare.framework.algorithm import Model
from flare.framework.computation_task import ComputationTask
from flare.algorithm_zoo.simple_algorithms import SimpleAC, SimpleQ
from flare.model_zoo.simple_models import SimpleModelDeterministic, SimpleModelAC, SimpleModelQ
from test_algorithm import TestAlgorithm

from torch.distributions import Categorical
import torch.nn as nn
import torch

import numpy as np
from copy import deepcopy
import unittest
import math


class TestModelCNN(Model):
    def __init__(self, width, height, num_actions):
        super(TestModelCNN, self).__init__()
        self.img_channels = 1
        self.height = height
        self.width = width

        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=self.img_channels,
                out_channels=1,
                kernel_size=3,
                bias=False),
            nn.ReLU())
        self.net = nn.Sequential(
            nn.Linear(
                (self.height - 2) * (self.width - 2), 32, bias=False),
            nn.ReLU(),
            nn.Linear(
                32, 16, bias=False),
            nn.ReLU(),
            nn.Linear(
                16, num_actions, bias=False),
            nn.Softmax(dim=1))

    def get_input_specs(self):
        ## image format CHW
        return [("image",
                 dict(shape=[self.img_channels, self.height, self.width]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def policy(self, inputs, states):
        cnn_out = self.cnn(inputs.values()[0])
        cnn_out = cnn_out.view(-1, (self.height - 2) * (self.width - 2))
        dist = Categorical(self.net(cnn_out))
        return dict(action=dist), states


class TestComputationTask(unittest.TestCase):
    def test_predict(self):
        """
        Test case for AC-learning and Q-learning predictions
        """
        num_actions = 4

        def test(input, ct, max):
            action_counter = [0] * num_actions
            total = 2000
            for i in range(total):
                actions, states = ct.predict(inputs=input)
                assert not states, "states should be empty"
                ## actions["action"] is a batch of actions
                for a in actions["action"]:
                    action_counter[a[0]] += 1

            if max:
                ### if max, the first action will always be chosen
                for i in range(num_actions):
                    prob = action_counter[i] / float(sum(action_counter))
                    self.assertAlmostEqual(
                        prob, 1.0 if i == 0 else 0.0, places=1)
            else:
                ### the actions should be uniform
                for i in range(num_actions):
                    prob = action_counter[i] / float(sum(action_counter))
                    self.assertAlmostEqual(prob, 1.0 / num_actions, places=1)

        dims = 100

        q_cnn = SimpleQ(model=TestModelCNN(
            width=84, height=84, num_actions=num_actions))

        q = SimpleQ(model=SimpleModelQ(
            dims=dims,
            num_actions=num_actions,
            mlp=nn.Sequential(
                nn.Linear(
                    dims, 32, bias=False),
                nn.ReLU(),
                nn.Linear(
                    32, 16, bias=False),
                nn.ReLU(),
                nn.Linear(
                    16, num_actions, bias=False))))

        batch_size = 10
        height, width = 84, 84
        sensor = np.zeros([batch_size, dims]).astype("float32")
        image = np.zeros([batch_size, 1, height, width]).astype("float32")

        ct0 = ComputationTask("test", algorithm=q_cnn)
        ct1 = ComputationTask("test", algorithm=q)

        test(dict(image=image), ct0, max=False)
        test(dict(sensor=sensor), ct1, max=True)

    def test_ct_para_sharing(self):
        """
        Test case for two CTs sharing parameters
        """
        alg = TestAlgorithm(model=SimpleModelDeterministic(
            dims=10, mlp=nn.Linear(10, 10)))
        ct0 = ComputationTask("test", algorithm=alg)
        ct1 = ComputationTask("test", algorithm=alg)

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, alg.model.dims]).astype("float32")

        outputs0, _ = ct0.predict(inputs=dict(sensor=sensor))
        outputs1, _ = ct1.predict(inputs=dict(sensor=sensor))
        self.assertEqual(
            np.sum(outputs0["continuous_action"].flatten()),
            np.sum(outputs1["continuous_action"].flatten()))

    def test_ct_para_copy(self):
        """
        Test case for two CTs copying parameters
        """

        alg = TestAlgorithm(model=SimpleModelDeterministic(
            dims=10, mlp=nn.Linear(10, 10)))

        ct0 = ComputationTask("test", algorithm=alg)
        ct1 = ComputationTask("test", algorithm=deepcopy(alg))

        batch_size = 10
        sensor = np.random.uniform(
            0, 1, [batch_size, ct0.alg.model.dims]).astype("float32")

        outputs0, _ = ct0.predict(inputs=dict(sensor=sensor))
        outputs1, _ = ct1.predict(inputs=dict(sensor=sensor))
        self.assertEqual(
            np.sum(outputs0["continuous_action"].flatten()),
            np.sum(outputs1["continuous_action"].flatten()))

    def test_ct_learning(self):
        """
        Test training
        """
        num_actions = 2
        dims = 100
        batch_size = 8
        sensor = np.ones(
            [batch_size, dims]).astype("float32") / dims  # normalize
        next_sensor = np.zeros([batch_size, dims]).astype("float32")

        for on_policy in [True, False]:
            if on_policy:
                alg = SimpleAC(model=SimpleModelAC(
                    dims=dims,
                    num_actions=num_actions,
                    mlp=nn.Sequential(
                        nn.Linear(
                            dims, 64, bias=False),
                        nn.ReLU(),
                        nn.Linear(
                            64, 32, bias=False),
                        nn.ReLU())))
                ct = ComputationTask(
                    "test", algorithm=alg, hyperparas=dict(lr=1e-1))
            else:
                alg = SimpleQ(
                    model=SimpleModelQ(
                        dims=dims,
                        num_actions=num_actions,
                        mlp=nn.Sequential(
                            nn.Linear(
                                dims, 64, bias=False),
                            nn.ReLU(),
                            nn.Linear(
                                64, 32, bias=False),
                            nn.ReLU(),
                            nn.Linear(
                                32, num_actions, bias=False))),
                    update_ref_interval=100)
                ct = ComputationTask(
                    "test", algorithm=alg, hyperparas=dict(lr=1e-1))

            for i in range(1000):
                if on_policy:
                    outputs, _ = ct.predict(inputs=dict(sensor=sensor))
                    actions = outputs["action"]
                else:
                    ## randomly assemble a batch
                    actions = np.random.choice(
                        [0, 1], size=(batch_size, 1),
                        p=[0.5, 0.5]).astype("int")
                rewards = (1.0 - actions).astype("float32")
                cost = ct.learn(
                    inputs=dict(sensor=sensor),
                    next_inputs=dict(sensor=next_sensor),
                    next_episode_end=dict(episode_end=np.ones(
                        (batch_size, 1)).astype("float32")),
                    actions=dict(action=actions),
                    rewards=dict(reward=rewards))

            ### the policy should bias towards the first action
            outputs, _ = ct.predict(inputs=dict(sensor=sensor))
            for a in outputs["action"]:
                self.assertEqual(a[0], 0)


if __name__ == "__main__":
    unittest.main()
