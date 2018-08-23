import torch.nn as nn
from flare.algorithm_zoo.simple_algorithms import SimpleAC
from flare.framework.manager import Manager
from flare.model_zoo.simple_models import SimpleModelAC
from flare.agent_zoo.simple_rl_agents import SimpleRLAgent
from flare.framework.agent import OnlineHelper
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
    mlp = nn.Sequential(
        nn.Linear(state_shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())

    alg = SimpleAC(model=SimpleModelAC(
        dims=state_shape, num_actions=num_actions, perception_net=mlp))

    # 3. Specify the settings for learning: data sampling strategy
    # (OnlineHelper here) and other settings used by
    # ComputationTask.
    ct_settings = {
        "RL": dict(
            algorithm=alg,
            hyperparas=dict(lr=5e-5),
            # sampling
            agent_helper=OnlineHelper,
            # each agent will call `learn()` every `sample_interval` steps
            sample_interval=4,
            num_agents=num_agents)
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
