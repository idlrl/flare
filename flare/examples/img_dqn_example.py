import torch.nn as nn
import numpy as np
import torch.optim as optim
from flare.algorithm_zoo.simple_algorithms import SimpleQ
from flare.model_zoo.simple_models import SimpleModelQ
from flare.framework.manager import Manager
from flare.agent_zoo.simple_rl_agents import SimpleRLAgent
from flare.framework.agent import ExpReplayHelper
from flare.env_zoo.gym_env import GymEnvImage
from flare.framework.common_functions import Flatten

if __name__ == '__main__':
    """
    A demo of how to train from image inputs
    """
    game = "Breakout-v0"

    num_agents = 16
    num_games = 8000

    im_height, im_width = 84, 84
    env = GymEnvImage(
        game, contexts=4, height=im_height, width=im_width, gray=True)
    d, h, w = env.observation_dims()["sensor"]
    num_actions = env.action_dims()["action"]

    # 1. Spawn one agent for each instance of environment.
    #    Agent's behavior depends on the actual algorithm being used. Since we
    #    are using SimpleAC, a proper type of Agent is SimpleRLAgent.
    agents = []
    for _ in range(num_agents):
        agent = SimpleRLAgent(num_games, reward_shaping_f=np.sign)
        agent.set_env(
            GymEnvImage,
            game_name=game,
            contexts=4,
            height=im_height,
            width=im_width,
            gray=True)
        agents.append(agent)

    # 2. Construct the network and specify the algorithm.
    #    Here we use a small CNN as the perception net for the Actor-Critic algorithm
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

    alg = SimpleQ(
        model=SimpleModelQ(
            dims=(d, h, w), num_actions=num_actions, perception_net=cnn),
        gpu_id=0,
        optim=(optim.RMSprop, dict(lr=1e-4)),
        grad_clip=5.0,
        exploration_end_steps=500000 // num_agents,
        update_ref_interval=100)

    # 3. Specify the settings for learning: data sampling strategy
    # (ExpReplayHelper here) and other settings used by
    # ComputationTask.
    ct_settings = {
        "RL": dict(
            alg=alg,
            # sampling
            agent_helper=ExpReplayHelper,
            buffer_capacity=200000 // num_agents,
            num_experiences=4,  # num per agent
            num_seqs=0,  # sample instances
            sample_interval=5)
    }

    # 4. Create Manager that handles the running of the whole pipeline
    manager = Manager(ct_settings)
    manager.add_agents(agents)
    manager.start()
