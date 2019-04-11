#!/usr/bin/python

from __future__ import print_function
import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import numpy as np
import sys
sys.path.append("..")

from flare.model_zoo.simple_models import GaussianPolicyModel
from flare.algorithm_zoo.simple_algorithms import OffPolicyAC
from flare.framework.manager import Manager
from flare.framework.agent import OnlineHelper
from flare.env_zoo.gym_env import GymEnv
from flare.agent_zoo.simple_rl_agents import SimpleRLAgent


class ContinuousTask(object):
    def make_model(self, args):
        env = GymEnv(args.game)
        num_dims = env.observation_dims()["sensor"]
        action_dims = env.action_dims()["action"]

        print("Input dims: {}".format(num_dims))
        print("Action dims: {}".format(action_dims))
        print("Action space: {}".format(env.action_range()["action"]))

        hidden_size = 256
        mlp = nn.Sequential(
            nn.Linear(num_dims[0], hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU())
        model = GaussianPolicyModel(num_dims,
                                    action_dims,
                                    perception_net=mlp)
        return model

    def run(self, args):
        model = self.make_model(args)
        opt = optim.RMSprop(model.parameters(), lr=args.lr)
        alg = OffPolicyAC(
            model=model,
            optim=opt,
            epsilon=0.2,
            prob_entropy_weight=args.entropy_w,
            gpu_id=args.gpu)

        ct_settings = {
            "RL": dict(
                alg=alg,
                # sampling
                agent_helper=OnlineHelper,
                agents_per_batch=args.agents_per_batch,
                # each agent will call `learn()` every `sample_interval` steps
                sample_interval=args.history_len)
        }

        log_settings = dict(
            print_interval=args.log_interval)

        reward_shaping_f = lambda x: x / 100
        agents = []
        for _ in range(args.num_agents):
            agent = SimpleRLAgent(args.num_games,
                                  reward_shaping_f=reward_shaping_f)
            agent.set_env(GymEnv, game_name=args.game)
            agents.append(agent)

        # 4. Create Manager that handles the running of the whole pipeline
        manager = Manager(ct_settings, log_settings)
        manager.add_agents(agents)
        manager.start()

        # 5. compute last reward
        return np.mean(manager.stats['All'].data_q['total_reward'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OffPolicy-AC continuous cases')

    parser.add_argument('--entropy_w', type=float, default=1e-4,
                        help="Entropy cost weight")
    parser.add_argument('--num_games', type=int, default=20,
                        help='number of games')
    parser.add_argument('--game', type=str, default='BipedalWalker-v2', help='Name of the game')

    parser.add_argument('--history_len', type=int, default=2, metavar='N',
                        help='length of history for training (default: 4)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu id (-1 for cpu)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    # total # of games = num_agents * num_games
    parser.add_argument('--agents_per_batch', type=int, default=64,
                        help='Each batch contains data from X agents')
    parser.add_argument('--num_agents', type=int, default=512,
                        help='Number of parallel agents')
    parser.add_argument('--log_interval', type=int, default=200, metavar='N',
                        help='how many games to wait before logging training status')
    args = parser.parse_args()

    task = ContinuousTask()

    #### Continuous control training examples                        ####
    #### Each game should have a decent score by the end of training ####

    # BipedalWalker-v2
    task.run(args)

    if False: # set True to run more examples
        # Ant-v2
        args.num_games = 60
        args.game = "Ant-v2"
        task.run(args)

        # Hopper-v2
        args.num_games = 300
        args.game = "Hopper-v2"
        args.entropy_w = 5e-3
        task.run(args)

        # Walker2d-v2
        args.num_games = 400
        args.game = "Walker2d-v2"
        args.entropy_w = 5e-3
        task.run(args)
