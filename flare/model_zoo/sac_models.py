from flare.model_zoo.simple_models import SimpleModelAC
import torch.nn as nn


class SACModel(SimpleModelAC):
    def __init__(self, dims, num_actions, perception_net):
        super(SACModel, self).__init__(dims, num_actions, perception_net)
        hidden_size = list(perception_net.modules())[-2].out_features
        self.q_net = nn.Sequential(perception_net,
                                   nn.Linear(hidden_size, num_actions))

    def value(self, inputs, states):
        d, states = super(SACModel, self).value(inputs, states)
        d["q_value"] = self.q_net(inputs.values()[0])
        return d, states
