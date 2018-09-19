import torch.nn as nn
import torch.optim as optim
from flare.algorithm_zoo.simple_algorithms import SimpleAC
from flare.framework.manager import Manager
from flare.model_zoo.simple_models import SimpleModelAC
from flare.agent_zoo.simple_rl_agents import SimpleRLAgent
from flare.framework.agent import OnlineHelper
from flare.env_zoo.gym_env import GymEnv

if __name__ == '__main__':
    """
    A demo of how to run a simple RL experiment
    """
    game = "CartPole-v0"

    num_agents = 16
    num_games = 8000

    env = GymEnv(game)
    state_shape = env.observation_dims()["sensor"]
    num_actions = env.action_dims()["action"]

    # 1. Spawn one agent for each instance of environment.
    #    Agent's behavior depends on the actual algorithm being used. Since we
    #    are using SimpleAC, a proper type of Agent is SimpleRLAgent.
    reward_shaping_f = lambda x: x / 100.0
    agents = []
    for _ in range(num_agents):
        agent = SimpleRLAgent(num_games, reward_shaping_f=reward_shaping_f)
        agent.set_env(GymEnv, game_name=game)
        agents.append(agent)

    # 2. Construct the network and specify the algorithm.
    #    Here we use a small MLP and apply the Actor-Critic algorithm
    mlp = nn.Sequential(
        nn.Linear(state_shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())

    alg = SimpleAC(
        model=SimpleModelAC(
            dims=state_shape, num_actions=num_actions, perception_net=mlp),
        optim=(optim.RMSprop, dict(lr=5e-5)),
        gpu_id=-1)  ## use cpu

    # 3. Specify the settings for learning: data sampling strategy
    # (OnlineHelper here) and other settings used by
    # ComputationTask.
    ct_settings = {
        "RL": dict(
            algorithm=alg,
            # sampling
            agent_helper=OnlineHelper,
            # each agent will call `learn()` every `sample_interval` steps
            sample_interval=4,
            num_agents=num_agents)
    }

    # 4. Create Manager that handles the running of the whole pipeline
    manager = Manager(ct_settings)
    manager.add_agents(agents)
    manager.start()
