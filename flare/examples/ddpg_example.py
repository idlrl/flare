import torch.optim as optim
from flare.agent_zoo.simple_rl_agents import ActionNoiseAgent
from flare.algorithm_zoo.ddpg import ContinuousDeterministicModel, DDPG
from flare.common.noises import OUNoise
from flare.framework.agent import ExpReplayHelper
from flare.framework.env import GymEnv
from flare.framework.manager import Manager

if __name__ == '__main__':
    """
    A demo of how to run a simple RL experiment
    """
    game = "Pendulum-v0"

    num_agents = 1
    num_games = 10000
    # 1. Create environments
    envs = []
    for _ in range(num_agents):
        envs.append(GymEnv(game))
    state_shape = envs[-1].observation_space.shape[0]
    action_space = envs[-1].action_space
    action_dims = action_space.shape[0]

    alg = DDPG(
        model=ContinuousDeterministicModel(
            input_dims=state_shape, action_dims=action_space.shape[0]),
        update_ref_interval=1,
        update_weight=0.001,
        # DDPG requires different optimizer settings for policy and critic.
        # In this example, we use Adam with learning rate 1e-4 for policy, and
        # Adm with learning rate 1e-3 and l2 weight decay 1e-2 for critic.
        policy_optim=(optim.Adam, dict(lr=1e-4)),
        critic_optim=(optim.Adam, dict(
            lr=1e-3, weight_decay=1e-2)))

    # 3. Specify the settings for learning: the algorithm to use (DDPG 
    # in this case), data sampling strategy (OnPolicyHelper here) and other 
    # settings used by ComputationTask.
    ct_settings = {
        "RL": dict(
            num_agents=num_agents,
            algorithm=alg,
            # sampling
            agent_helper=ExpReplayHelper,
            buffer_capacity=1000000 / num_agents,
            num_experiences=64,
            sample_interval=8,
            num_seqs=0)
    }

    # 4. Create Manager that handles the running of the whole framework
    manager = Manager(ct_settings)

    # 5. Spawn one agent for each instance of environment. 
    #    Agent's behavior depends on the actual algorithm being used. Since we 
    #    are using DDPG, a proper type of Agent is ActionNoiseAgent.
    for env in envs:
        agent = ActionNoiseAgent(env, num_games, OUNoise(action_dims))
        # An Agent has to be added into the Manager before we can use it to
        # interact with environment and collect data
        manager.add_agent(agent)

    manager.start()
