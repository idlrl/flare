#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0 #
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym
from random import randint
import torch.nn as nn
from flare.algorithm_zoo.simple_algorithms import SimpleAC
from flare.framework.manager import Manager
from flare.model_zoo.simple_models import SimpleRNNModelAC
from parl.agent_zoo.simple_rl_agents import SimpleRNNRLAgent
from parl.framework.agent import OnPolicyHelper

if __name__ == '__main__':
    """
    A demo of how to run a simple RL experiment
    """
    game = "CartPole-v0"

    num_agents = 4
    num_games = 8000
    # 1. Create environments
    envs = []
    for _ in range(num_agents):
        envs.append(gym.make(game))
    state_shape = envs[-1].observation_space.shape[0]
    num_actions = envs[-1].action_space.n

    # 2. Construct the network and specify the algorithm.
    #    Here we use a small MLP and apply the Actor-Critic algorithm
    hidden_size = 128
    mlp = nn.Sequential(
        nn.Linear(state_shape, hidden_size),
        nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())

    alg = SimpleAC(model=SimpleRNNModelAC(
        dims=state_shape, num_actions=num_actions, mlp=mlp))

    # 3. Specify the settings for learning: the algorithm to use (SimpleAC
    # in this case), data sampling strategy (OnPolicyHelper here) and other
    # settings used by ComputationTask.
    ct_settings = {
        "RL": dict(
            algorithm=alg,
            hyperparas=dict(lr=1e-4),
            # sampling
            agent_helper=OnPolicyHelper,
            sample_interval=8,
            num_agents=num_agents)
    }

    # 4. Create Manager that handles the running of the whole framework
    manager = Manager(ct_settings)

    # 5. Spawn one agent for each instance of environment.
    #    Agent's behavior depends on the actual algorithm being used. Since we
    #    are using SimpleAC, a proper type of Agent is SimpleRLAgent.
    for env in envs:
        agent = SimpleRNNRLAgent(env, num_games)
        # An Agent has to be added into the Manager before we can use it to
        # interact with environment and collect data
        manager.add_agent(agent)

    manager.start()
