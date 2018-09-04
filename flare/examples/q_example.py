from random import randint
import torch.nn as nn
from flare.algorithm_zoo.simple_algorithms import SimpleQ
from flare.framework.manager import Manager
from flare.model_zoo.simple_models import SimpleModelQ
from flare.framework.agent import ExpReplayHelper
from flare.env_zoo.gym_env import GymEnv
from flare.agent_zoo.simple_rl_agents import SimpleRLAgent

if __name__ == '__main__':
    """
    A demo of how to run a simple RL experiment
    """
    game = "MountainCar-v0"

    num_agents = 16
    num_games = 8000
    # 1. Create environments
    envs = []
    for _ in range(num_agents):
        envs.append(GymEnv(game))
    state_shape = envs[-1].observation_dims()[0]
    num_actions = envs[-1].action_dims()[0]

    # 2. Construct the network and specify the algorithm.
    #    Here we use a small MLP and apply the Q-learning algorithm
    inner_size = 256
    mlp = nn.Sequential(
        nn.Linear(state_shape[0], inner_size),
        nn.ReLU(),
        nn.Linear(inner_size, inner_size),
        nn.ReLU(), nn.Linear(inner_size, inner_size), nn.ReLU())

    alg = SimpleQ(
        model=SimpleModelQ(
            dims=state_shape, num_actions=num_actions, perception_net=mlp),
        exploration_end_steps=500000 / num_agents,
        update_ref_interval=100)

    # 3. Specify the settings for learning: the algorithm to use (SimpleAC
    # in this case), data sampling strategy (ExpReplayHelper here) and other
    # settings used by ComputationTask.
    ct_settings = {
        "RL": dict(
            num_agents=num_agents,
            algorithm=alg,
            hyperparas=dict(lr=1e-4),
            # sampling
            agent_helper=ExpReplayHelper,
            buffer_capacity=200000 / num_agents,
            num_experiences=4,  # num per agent
            num_seqs=0,  # sample instances
            sample_interval=8)
    }

    # 4. Create Manager that handles the running of the whole pipeline
    manager = Manager(ct_settings)

    # 5. Spawn one agent for each instance of environment.
    #    Agent's behavior depends on the actual algorithm being used. Since we
    #    are using SimpleAC, a proper type of Agent is SimpleRLAgent.
    reward_shaping_f = lambda x: x / 100.0
    for env in envs:
        agent = SimpleRLAgent(env, num_games, reward_shaping_f)
        # An Agent has to be added into the Manager before we can use it to
        # interact with environment and collect data
        manager.add_agent(agent)

    manager.start()
