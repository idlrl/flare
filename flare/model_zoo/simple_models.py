from flare.framework.algorithm import Model
from flare.common.distributions import Deterministic
from flare.common import common_functions as comf
import torch.nn as nn
from torch.distributions import Categorical


class SimpleModelDeterministic(Model):
    def __init__(self, dims, mlp):
        super(SimpleModelDeterministic, self).__init__()
        self.dims = dims
        self.mlp = mlp

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("continuous_action", dict(shape=[self.dims]))]

    def policy(self, inputs, states):
        hidden = self.mlp(inputs.values()[0])
        return dict(continuous_action=Deterministic(hidden)), states


class SimpleModelAC(Model):
    def __init__(self, dims, num_actions, mlp):
        super(SimpleModelAC, self).__init__()
        self.dims = dims
        hidden_size = list(mlp.modules())[-2].out_features
        self.policy_net = nn.Sequential(mlp,
                                        nn.Linear(hidden_size, num_actions),
                                        nn.Softmax())
        self.value_net = nn.Sequential(mlp, nn.Linear(hidden_size, 1))

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def policy(self, inputs, states):
        dist = Categorical(probs=self.policy_net(inputs.values()[0]))
        return dict(action=dist), states

    def value(self, inputs, states):
        return dict(v_value=self.value_net(inputs.values()[0]))


class SimpleModelQ(Model):
    def __init__(self, dims, num_actions, mlp):
        super(SimpleModelQ, self).__init__()
        self.dims = dims
        self.num_actions = num_actions
        self.mlp = mlp

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def policy(self, inputs, states):
        values = self.value(inputs, states)
        q_value = values["q_value"]
        return dict(action=comf.q_categorical(q_value)), states

    def value(self, inputs, states):
        return dict(q_value=self.mlp(inputs.values()[0]))
