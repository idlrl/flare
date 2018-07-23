import gym
from random import randint
import torch.nn as nn
from flare.algorithm_zoo.simple_algorithms import SimpleAC
from flare.framework.manager import Manager
from flare.model_zoo.simple_models import SimpleModelAC
from parl.agent_zoo.simple_rl_agents import SimpleRLAgent
from parl.framework.agent import OnPolicyHelper

if __name__ == '__main__':
    """
    A demo of how to run a simple RL experiment
    """
    game = "CartPole-v0"

    num_agent = 1
    num_games = 8000
    # 1. Create environments
    envs = []
    for _ in range(num_agent):
        envs.append(gym.make(game))
    state_shape = envs[-1].observation_space.shape[0]
    num_actions = envs[-1].action_space.n

    # 2. Construct the network and specify the algorithm.
    #    Here we use a small MLP and apply the Actor-Critic algorithm
    mlp = nn.Sequential(
        nn.Linear(state_shape, 128),
        nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())

    alg = SimpleAC(model=SimpleModelAC(
        dims=state_shape, num_actions=num_actions, mlp=mlp))

    # 3. Specify the settings for learning: the algorithm to use (SimpleAC 
    # in this case), data sampling strategy (OnPolicyHelper here) and other 
    # settings used by ComputationTask.
    ct_settings = {
        "RL": dict(
            algorithm=alg,
            hyperparas=dict(lr=1e-4),
            # sampling
            sample_method=OnPolicyHelper,
            sample_interval=8,
            # ct wrapper
            min_batchsize=1,
            max_batchsize=16,
            comm_timeout=1)
    }

    # 4. Create Manager that handles the running of the whole framework
    manager = Manager(ct_settings)

    # 5. Spawn one agent for each instance of environment. 
    #    Agent's behavior depends on the actual algorithm being used. Since we 
    #    are using SimpleAC, a proper type of Agent is SimpleRLAgent.
    for env in envs:
        agent = SimpleRLAgent(env, num_games)
        # An Agent has to be added into the Manager before we can use it to
        # interact with environment and collect data
        manager.add_agent(agent)

    manager.start()
