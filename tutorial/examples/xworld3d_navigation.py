import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from flare.framework.algorithm import Model
from flare.algorithm_zoo.simple_algorithms import SimpleAC
from flare.framework.manager import Manager
from flare.agent_zoo.simple_rl_agents import SimpleRNNRLAgent
from flare.framework.agent import OnlineHelper
from flare.env_zoo.xworld import XWorldEnv
from flare.framework.common_functions import BoW, Flatten


class GFT(nn.Module):
    """
    This class implements the GFT model proposed in the CoRL2018 paper:
    https://arxiv.org/pdf/1805.08329.pdf
    """

    def __init__(self, K, vision_perception_net, language_perception_net,
                 hidden_net):
        super(GFT, self).__init__()
        self.vision_perception_net = vision_perception_net
        self.language_perception_net = language_perception_net
        self.hidden_net = hidden_net
        self.channels = list(vision_perception_net.modules())[-2].out_channels
        ## We should use ModuleList, otherwise the parameters are not registered to GFT
        self.t_layers = nn.ModuleList([
            nn.Linear(language_perception_net.dim,
                      (1 + self.channels) * self.channels) for k in range(K)
        ])

    def forward(self, screen, sentence):
        sentence_embedding = self.language_perception_net(sentence)
        cnn_out = self.vision_perception_net(screen)
        cnn_out = cnn_out.view(cnn_out.size()[0], self.channels, -1)
        ## compute K transformation matrices
        ts = [l(sentence_embedding).view(-1, self.channels, (self.channels + 1)) \
              for l in self.t_layers]

        ones = torch.ones(cnn_out.size()[0], 1, cnn_out.size()[-1])
        if cnn_out.is_cuda:
            ones = ones.to(cnn_out.get_device())

        for t in ts:
            assert t.size()[0] == cnn_out.size()[0]
            cnn_out = torch.cat((cnn_out, ones), dim=1)
            cnn_out = F.relu(torch.matmul(t, cnn_out))
        return self.hidden_net(cnn_out)


class GFTModelAC(Model):
    def __init__(self,
                 img_dims,
                 num_actions,
                 vision_perception_net,
                 language_perception_net,
                 hidden_net,
                 K=2):
        super(GFTModelAC, self).__init__()
        assert isinstance(img_dims, list) or isinstance(img_dims, tuple)
        self.img_dims = img_dims
        self.hidden_size = list(hidden_net.modules())[-2].out_features
        self.gft = GFT(K, vision_perception_net, language_perception_net,
                       hidden_net)
        ## Two-layer RNN
        self.action_embedding = nn.Embedding(num_actions, self.hidden_size / 2)
        self.h_m_cell = nn.RNNCell(
            self.hidden_size, self.hidden_size, nonlinearity='relu')
        self.h_a_cell = nn.RNNCell(
            self.hidden_size / 2, self.hidden_size / 2, nonlinearity='relu')
        self.f_cell = nn.RNNCell(
            self.hidden_size, self.hidden_size, nonlinearity='relu')
        self.fc = nn.Sequential(
            nn.Linear(int(1.5 * self.hidden_size), self.hidden_size),
            nn.ReLU())
        self.policy_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, num_actions),
            nn.Softmax(dim=1))
        self.value_layer = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(), nn.Linear(self.hidden_size, 1))

    def _two_layer_recurrent(self, inputs, states):
        fusion = self.gft(inputs["screen"], inputs["sentence"])
        prev_action = inputs["prev_action"]
        h_m, h_a, f = states["h_m"], states["h_a"], states["f"]
        h_m_ = self.h_m_cell(fusion, h_m)
        h_a_ = self.h_a_cell(
            self.action_embedding(prev_action.squeeze(-1)), h_a)
        f_ = self.f_cell(self.fc(torch.cat((h_m_, h_a_), dim=1)), f)
        return h_m_, h_a_, f_

    def get_input_specs(self):
        return [("screen", dict(shape=self.img_dims)), ("sentence", dict(
            shape=[1], dtype="int64")), ("prev_action", dict(
                shape=[1], dtype="int64"))]

    def get_action_specs(self):
        return [("action", dict(shape=[1], dtype="int64"))]

    def get_state_specs(self):
        return [("h_m", dict(shape=[self.hidden_size])),
                ("h_a", dict(shape=[self.hidden_size / 2])),
                ("f", dict(shape=[self.hidden_size]))]

    def policy(self, inputs, states):
        h_m, h_a, f = self._two_layer_recurrent(inputs, states)
        dist = Categorical(probs=self.policy_layer(f))
        return dict(action=dist), dict(h_m=h_m, h_a=h_a, f=f)

    def value(self, inputs, states):
        h_m, h_a, f = self._two_layer_recurrent(inputs, states)
        v_value = self.value_layer(f)
        return dict(v_value=v_value), dict(h_m=h_m, h_a=h_a, f=f)


if __name__ == "__main__":
    num_agents = 26
    num_games = 1000000

    # 1. Create environment arguments
    im_size = 84
    options = {
        "x3_conf": "../../flare/env_zoo/tests/walls3d.json",
        "context": 1,
        "x3_training_img_width": im_size,
        "x3_training_img_height": im_size,
        "x3_turning_rad": 0.2,
        "curriculum": 0.7,
        "color": True,
        "pause_screen": True
    }

    with open("../../flare/env_zoo/tests/dict.txt") as f:
        word_list = [word.strip() for word in f.readlines()]

    env = XWorldEnv("xworld3d", options, word_list, opengl_init=False)
    d, h, w = env.observation_dims()["screen"]
    voc_size, = env.observation_dims()["sentence"]
    num_actions = env.action_dims()["action"]

    # 2. Spawn one agent for each instance of environment.
    #    Agent's behavior depends on the actual algorithm being used. Since we
    #    are using SimpleAC, a proper type of Agent is SimpleRLAgent.
    agents = []
    for _ in range(num_agents):
        agent = SimpleRNNRLAgent(num_games, learning=True, actrep=4)
        agent.set_env(
            XWorldEnv,
            game_name="xworld3d",
            options=options,
            word_list=word_list,
            show_frame=False)
        agents.append(agent)

    # 3. Construct the network and specify the algorithm.
    #    Here we use a small CNN as the perception net for the Actor-Critic algorithm,
    #    and use a BoW model to compute a sentence embedding
    cnn = nn.Sequential(
        nn.Conv2d(
            d, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(
            32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(
            64, 64, kernel_size=3, stride=1),
        nn.ReLU())

    word_embedding_dim = 128
    bow = BoW(dict_size=voc_size, dim=word_embedding_dim, std=0.1)
    hidden_size = 2 * word_embedding_dim

    hidden_net = nn.Sequential(Flatten(),
                               nn.Linear(7 * 7 * 64, hidden_size), nn.ReLU())

    alg = SimpleAC(
        model=GFTModelAC(
            img_dims=(d, h, w),
            num_actions=num_actions,
            vision_perception_net=cnn,
            language_perception_net=bow,
            hidden_net=hidden_net),
        gpu_id=1,
        value_cost_weight=1.0,
        prob_entropy_weight=0.05,
        optim=(optim.RMSprop, dict(
            lr=1e-5, momentum=0.9)),
        ntd=True)

    # 4. Specify the settings for learning: data sampling strategy
    # and other settings used by ComputationTask.
    ct_settings = {
        "RL": dict(
            algorithm=alg,
            show_para_every_backwards=500,
            # sampling
            agent_helper=OnlineHelper,
            # each agent will call `learn()` every `sample_interval` steps
            sample_interval=4,
            num_agents=num_agents)
    }

    log_settings = dict(
        model_dir="/tmp/test",
        print_interval=100,
        model_save_interval=20,
        load_model=False,
        pass_num=0,
        log_file="/tmp/log.txt")

    # 5. Create Manager that handles the running of the whole pipeline
    manager = Manager(ct_settings, log_settings)
    # An Agent has to be added into the Manager before we can use it to
    # interact with environment and collect data
    manager.add_agents(agents)
    manager.start()
