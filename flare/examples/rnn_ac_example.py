from random import randint
import torch.nn as nn
from flare.algorithm_zoo.simple_algorithms import SimpleAC
from flare.framework.manager import Manager
from flare.model_zoo.simple_models import SimpleRNNModelAC
from flare.agent_zoo.simple_rl_agents import SimpleRNNRLAgent
from flare.framework.agent import OnPolicyHelper
from flare.framework.env import GymEnv

if __name__ == '__main__':
    """
    A demo of how to run a simple RL experiment
    """
    game = "CartPole-v0"

    num_agents = 16
    num_games = 8000
    # 1. Create environments
    envs = []
    for _ in range(num_agents):
        envs.append(GymEnv(game))
    state_shape = envs[-1].observation_dims()[0]
    num_actions = envs[-1].action_dims()[0]

    # 2. Construct the network and specify the algorithm.
    #    Here we use a small MLP and apply the Actor-Critic algorithm
    hidden_size = 128
    mlp = nn.Sequential(
        nn.Linear(state_shape[0], hidden_size),
        nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU())

    alg = SimpleAC(model=SimpleRNNModelAC(
        dims=state_shape, num_actions=num_actions, perception_net=mlp))

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

    # 4. Create Manager that handles the running of the whole pipeline
    manager = Manager(ct_settings)

    # 5. Spawn one agent for each instance of environment.
    #    Agent's behavior depends on the actual algorithm being used. Since we
    #    are using SimpleAC, a proper type of Agent is SimpleRNNRLAgent.
    reward_shaping_f = lambda x: x / 100.0
    for env in envs:
        agent = SimpleRNNRLAgent(env, num_games, reward_shaping_f)
        # An Agent has to be added into the Manager before we can use it to
        # interact with environment and collect data
        manager.add_agent(agent)

    manager.start()
