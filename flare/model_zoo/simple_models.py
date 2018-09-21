from flare.common.distributions import Deterministic
from flare.framework.algorithm import Model
from flare.framework import common_functions as comf
import torch
import torch.nn as nn
from torch.distributions import Categorical, MultivariateNormal


class SimpleModelDeterministic(Model):
    def __init__(self, dims, perception_net):
        super(SimpleModelDeterministic, self).__init__()
        assert isinstance(dims, list) or isinstance(dims, tuple)
        self.dims = dims
        self.perception_net = perception_net

    def get_input_specs(self):
        return [("sensor", dict(shape=self.dims))]

    def get_action_specs(self):
        return [("action", dict(shape=self.dims))]

    def policy(self, inputs, states):
        hidden = self.perception_net(inputs.values()[0])
        return dict(action=Deterministic(hidden)), states


class SimpleModelAC(Model):
    def __init__(self, dims, num_actions, perception_net):
        super(SimpleModelAC, self).__init__()
        assert isinstance(dims, list) or isinstance(dims, tuple)
        self.dims = dims
        hidden_size = list(perception_net.modules())[-2].out_features
        self.policy_net = nn.Sequential(
            perception_net,
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=1))
        self.value_net = nn.Sequential(perception_net,
                                       nn.Linear(hidden_size, 1))

    def get_input_specs(self):
        return [("sensor", dict(shape=self.dims))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def policy(self, inputs, states):
        dist = Categorical(probs=self.policy_net(inputs.values()[0]))
        return dict(action=dist), states

    def value(self, inputs, states):
        return dict(v_value=self.value_net(inputs.values()[0])), states


class SimpleModelQ(Model):
    def __init__(self, dims, num_actions, perception_net):
        super(SimpleModelQ, self).__init__()
        assert isinstance(dims, list) or isinstance(dims, tuple)
        self.dims = dims
        self.num_actions = num_actions
        hidden_size = list(perception_net.modules())[-2].out_features
        self.value_net = nn.Sequential(perception_net,
                                       nn.Linear(hidden_size, num_actions))

    def get_input_specs(self):
        return [("sensor", dict(shape=self.dims))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def policy(self, inputs, states):
        values, states = self.value(inputs, states)
        q_value = values["q_value"]
        return dict(action=comf.q_categorical(q_value)), states

    def value(self, inputs, states):
        return dict(q_value=self.value_net(inputs.values()[0])), states


class SimpleRNNModelAC(Model):
    def __init__(self, dims, num_actions, perception_net):
        super(SimpleRNNModelAC, self).__init__()
        assert isinstance(dims, list) or isinstance(dims, tuple)
        self.dims = dims
        self.num_actions = num_actions
        self.hidden_size = list(perception_net.children())[-2].out_features
        self.hidden_layers = perception_net
        self.recurrent = nn.RNNCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            nonlinearity="relu")
        self.policy_layers = nn.Sequential(
            nn.Linear(self.hidden_size, num_actions), nn.Softmax(dim=1))
        self.value_layer = nn.Linear(self.hidden_size, 1)

    def get_input_specs(self):
        return [("sensor", dict(shape=self.dims))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def get_state_specs(self):
        return [("state", dict(shape=[self.hidden_size]))]

    def policy(self, inputs, states):
        hidden = self.hidden_layers(inputs.values()[0])
        next_state = self.recurrent(hidden, states.values()[0])
        dist = Categorical(probs=self.policy_layers(next_state))
        return dict(action=dist), dict(state=next_state)

    def value(self, inputs, states):
        hidden = self.hidden_layers(inputs.values()[0])
        next_state = self.recurrent(hidden, states.values()[0])
        return dict(v_value=self.value_layer(next_state)), dict(
            state=next_state)


class SimpleRNNModelQ(Model):
    def __init__(self, dims, num_actions, perception_net):
        super(SimpleRNNModelQ, self).__init__()
        assert isinstance(dims, list) or isinstance(dims, tuple)
        self.dims = dims
        self.num_actions = num_actions
        self.hidden_layers = perception_net
        self.hidden_size = list(perception_net.children())[-2].out_features
        self.value_layer = nn.Linear(self.hidden_size, num_actions)
        self.recurrent = nn.RNNCell(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            nonlinearity="relu")

    def get_input_specs(self):
        return [("sensor", dict(shape=self.dims))]

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
        return dict(q_value=self.value_layer(next_state)), dict(
            state=next_state)


class GaussianPolicyModel(Model):
    def __init__(self, dims, action_dims, perception_net, std=0.01):
        super(GaussianPolicyModel, self).__init__()
        assert isinstance(dims, list) or isinstance(dims, tuple)
        self.dims = dims
        self.action_dims = action_dims
        self.std = std
        hidden_size = list(perception_net.modules())[-2].out_features
        self.policy_net = nn.Sequential(perception_net,
                                        nn.Linear(hidden_size, action_dims))
        self.value_net = nn.Sequential(perception_net,
                                       nn.Linear(hidden_size, 1))

    def get_input_specs(self):
        return [("sensor", dict(shape=self.dims))]

    def get_action_specs(self):
        return [("action", dict(shape=[self.action_dims]))]

    def policy(self, inputs, states):
        dist = MultivariateNormal(
            loc=self.policy_net(inputs.values()[0]),
            covariance_matrix=torch.eye(self.action_dims) * self.std)
        return dict(action=dist), states

    def value(self, inputs, states):
        return dict(v_value=self.value_net(inputs.values()[0])), states
