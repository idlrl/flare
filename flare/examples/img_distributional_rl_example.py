import argparse
import torch.nn as nn
import numpy as np
from flare.algorithm_zoo.distributional_rl_algorithms import C51
from flare.model_zoo.distributional_rl_models import C51Model
from flare.algorithm_zoo.distributional_rl_algorithms import QRDQN
from flare.model_zoo.distributional_rl_models import QRDQNModel
from flare.algorithm_zoo.distributional_rl_algorithms import IQN
from flare.model_zoo.distributional_rl_models import IQNModel
from flare.framework.manager import Manager
from flare.agent_zoo.simple_rl_agents import SimpleRLAgent
from flare.framework.agent import ExpReplayHelper
from flare.env_zoo.gym_env import GymEnvImage
from flare.framework.common_functions import Flatten


def c51(cnn, dims, num_actions, num_agents):
    alg = C51(model=C51Model(
        dims=dims, num_actions=num_actions, perception_net=cnn),
              gpu_id=0,
              exploration_end_steps=500000 / num_agents,
              update_ref_interval=100)

    ct_settings = {
        "RL": dict(
            num_agents=num_agents,
            algorithm=alg,
            hyperparas=dict(grad_clip=5.0),
            # sampling
            agent_helper=ExpReplayHelper,
            buffer_capacity=200000 / num_agents,
            num_experiences=4,  # num per agent
            num_seqs=0,  # sample instances
            sample_interval=5)
    }
    return ct_settings


def rqdqn(cnn, dims, num_actions, num_agents):
    alg = QRDQN(
        model=QRDQNModel(
            dims=dims, num_actions=num_actions, perception_net=cnn),
        gpu_id=0,
        exploration_end_steps=500000 / num_agents,
        update_ref_interval=100)

    ct_settings = {
        "RL": dict(
            num_agents=num_agents,
            algorithm=alg,
            hyperparas=dict(grad_clip=5.0),
            # sampling
            agent_helper=ExpReplayHelper,
            buffer_capacity=200000 / num_agents,
            num_experiences=4,  # num per agent
            num_seqs=0,  # sample instances
            sample_interval=5)
    }
    return ct_settings


def iqn(cnn, dims, num_actions, num_agents):
    alg = IQN(model=IQNModel(
        dims=dims, num_actions=num_actions, perception_net=cnn),
              gpu_id=0,
              exploration_end_steps=500000 / num_agents,
              update_ref_interval=100)

    ct_settings = {
        "RL": dict(
            num_agents=num_agents,
            algorithm=alg,
            hyperparas=dict(grad_clip=5.0),
            # sampling
            agent_helper=ExpReplayHelper,
            buffer_capacity=200000 / num_agents,
            num_experiences=4,  # num per agent
            num_seqs=0,  # sample instances
            sample_interval=5)
    }
    return ct_settings


def get_settings(cnn, dims, num_actions, num_agents, name="C51"):
    if name == "C51":
        return c51(cnn, dims, num_actions, num_agents)
    elif name == "QR-DQN":
        return rqdqn(cnn, dims, num_actions, num_agents)
    elif name == "IQN":
        return iqn(cnn, dims, num_actions, num_agents)
    else:
        raise ValueError('algorithm name is not defined: ' + name)


def main(args):
    game = "Breakout-v0"

    num_agents = 16
    num_games = 8000

    # 1. Create image environments
    im_height, im_width = 84, 84
    envs = []
    for _ in range(num_agents):
        envs.append(
            GymEnvImage(
                game, contexts=4, height=im_height, width=im_width, gray=True))
    # context screens
    d, h, w = envs[-1].observation_dims()[0]
    num_actions = envs[-1].action_dims()[0]

    # 2. Construct the network and specify the algorithm.
    #    We use a CNN as the perception net for the Actor-Critic algorithm
    cnn = nn.Sequential(
        nn.Conv2d(
            d, 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(
            32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(
            64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        Flatten(),  # flatten the CNN cube to a vector
        nn.Linear(7 * 7 * 64, 512),
        nn.ReLU())

    # 3. Specify the algorithm and settings for learning.
    ct_settings = get_settings(
        cnn, (d, h, w), num_actions, num_agents, name=args.name)

    # 4. Create Manager that handles the running of the whole pipeline
    manager = Manager(ct_settings)

    # 5. Spawn one agent for each instance of environment.
    #    Agent's behavior depends on the actual algorithm being used.
    for env in envs:
        agent = SimpleRLAgent(env, num_games, reward_shaping_f=np.sign)
        # An Agent has to be added into the Manager before we can use it to
        # interact with environment and collect data
        manager.add_agent(agent)

    manager.start()


if __name__ == '__main__':
    """
    A demo of how to train from image inputs
    """
    parser = argparse.ArgumentParser(
        description="Distributional RL with image input.")
    parser.add_argument(
        '--name',
        type=str,
        default='C51',
        help='Algorithm name from "C51", "QR-DQN" and "IQN"')
    args = parser.parse_args()
    main(args)
