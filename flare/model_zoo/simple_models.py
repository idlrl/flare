from flare.framework.algorithm import Model
from flare.framework.distributions import Deterministic
from flare.framework import common_functions as comf
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal


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
        return dict(v_value=self.value_net(inputs.values()[0])), states


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
        values, states = self.value(inputs, states)
        q_value = values["q_value"]
        return dict(action=comf.q_categorical(q_value)), states

    def value(self, inputs, states):
        return dict(q_value=self.mlp(inputs.values()[0])), states


class SimpleRNNModelQ(Model):
    def __init__(self, dims, num_actions, mlp):
        super(SimpleRNNModelQ, self).__init__()
        self.dims = dims
        self.num_actions = num_actions
        self.hidden_layers = mlp
        self.hidden_size = list(mlp.children())[-2].out_features
        self.value_layer = nn.Linear(self.hidden_size, num_actions)
        self.recurrent = nn.RNNCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            nonlinearity="relu")

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def get_state_specs(self):
        return [("state", dict(shape=[self.hidden_size]))]

    def policy(self, inputs, states):
        values, next_states = self.value(inputs, states)
        q_value = values["q_value"]
        return dict(action=comf.q_categorical(q_value)), next_states

    def value(self, inputs, states):
        hidden = self.hidden_layers(inputs.values()[0])
        next_state = self.recurrent(hidden, states.values()[0])
        #        next_state = hidden
        return dict(q_value=self.value_layer(next_state)), dict(
            state=next_state)


class GaussianPolicyModel(Model):
    def __init__(self, dims, action_dims, mlp, std=0.01):
        super(GaussianPolicyModel, self).__init__()
        self.dims = dims
        self.action_dims = action_dims
        self.std = std
        hidden_size = list(mlp.modules())[-2].out_features
        self.policy_net = nn.Sequential(mlp,
                                        nn.Linear(hidden_size, action_dims))
        self.value_net = nn.Sequential(mlp, nn.Linear(hidden_size, 1))

    def get_input_specs(self):
        return [("sensor", dict(shape=[self.dims]))]

    def get_action_specs(self):
        return [("action", dict(shape=[self.action_dims]))]

    def policy(self, inputs, states):
        dist = MultivariateNormal(
            loc=self.policy_net(inputs.values()[0]),
            covariance_matrix=torch.eye(self.action_dims) * self.std)
        return dict(action=dist), states

    def value(self, inputs, states):
        return dict(v_value=self.value_net(inputs.values()[0])), states
