import torch.nn as nn
import torch.optim as optim
import numpy as np
from flare.algorithm_zoo.simple_algorithms import SimpleAC
from flare.model_zoo.simple_models import SimpleModelAC
from flare.framework.manager import Manager
from flare.agent_zoo.simple_rl_agents import SimpleRLAgent
from flare.framework.agent import OnlineHelper
from flare.env_zoo.gym_env import GymEnvImage
from flare.framework.common_functions import Flatten

if __name__ == '__main__':
    """
    A demo of how to train from image inputs
    """
    game = "Assault-v0"

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

    alg = SimpleAC(
        model=SimpleModelAC(dims=(d, h, w),
                            num_actions=num_actions,
                            perception_net=cnn),
        optim=(optim.RMSprop, dict(lr=1e-4)),
        grad_clip=5.0,
        gpu_id=1)

    # 3. Specify the settings for learning: data sampling strategy
    # (OnlineHelper here) and other settings used by
    # ComputationTask.
    ct_settings = {
        "RL": dict(
            algorithm=alg,
            # sampling
            agent_helper=OnlineHelper,
            # each agent will call `learn()` every `sample_interval` steps
            sample_interval=5,
            num_agents=num_agents)
    }

    # 4. Create Manager that handles the running of the whole pipeline
    manager = Manager(ct_settings)

    # 5. Spawn one agent for each instance of environment.
    #    Agent's behavior depends on the actual algorithm being used. Since we
    #    are using SimpleAC, a proper type of Agent is SimpleRLAgent.
    for env in envs:
        agent = SimpleRLAgent(env, num_games, reward_shaping_f=np.sign)
        # An Agent has to be added into the Manager before we can use it to
        # interact with environment and collect data
        manager.add_agent(agent)

    manager.start()
