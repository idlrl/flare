import torch.optim as optim
from flare.agent_zoo.simple_rl_agents import SimpleRLAgent
from flare.algorithm_zoo.ddpg import ContinuousDeterministicModel, DDPG
from flare.common.noises import OUNoise
from flare.framework.agent import ExpReplayHelper
from flare.env_zoo.gym_env import GymEnv
from flare.framework.manager import Manager


class ActionNoiseAgent(SimpleRLAgent):
    """
    This class extends `SimpleRLAgent` by applying action noise after
    prediction. It can be used to algorithms with deterministic policies, e.g.,
    `DDPG`.
    """

    def __init__(self,
                 num_games,
                 action_noise,
                 actrep=1,
                 learning=True,
                 reward_shaping_f=lambda x: x):
        super(ActionNoiseAgent, self).__init__(num_games, actrep, learning,
                                               reward_shaping_f)
        self.action_noise = action_noise

    def _cts_predict(self, observations, states):
        assert len(observations) == 1
        actions, _ = self.predict('RL', observations)
        assert len(actions) == 1
        actions = {
            k: a + self.action_noise.noise()
            for k, a in actions.iteritems()
        }
        return actions, dict()

    def _reset_env(self):
        self.action_noise.reset()
        return super(ActionNoiseAgent, self)._reset_env()


if __name__ == '__main__':
    """
    A demo of how to run a simple RL experiment
    """
    game = "Pendulum-v0"

    num_agents = 1
    num_games = 10000

    env = GymEnv(game)
    state_shape = env.observation_dims()["sensor"]
    action_dims = env.action_dims()["action"]

    # 1. Spawn one agent for each instance of environment.
    #    Agent's behavior depends on the actual algorithm being used. Since we
    #    are using DDPG, a proper type of Agent is ActionNoiseAgent.
    agents = []
    for _ in range(num_agents):
        agent = ActionNoiseAgent(num_games, OUNoise(action_dims))
        agent.set_env(GymEnv, game)
        agents.append(agent)

    alg = DDPG(
        model=ContinuousDeterministicModel(
            input_dims=state_shape[0], action_dims=action_dims),
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
            show_para_every_backwards=500,
            # sampling
            agent_helper=ExpReplayHelper,
            buffer_capacity=1000000 / num_agents,
            num_experiences=64,
            sample_interval=8,
            num_seqs=0)
    }

    # 4. Create Manager that handles the running of the whole framework
    manager = Manager(ct_settings)
    # An Agent has to be added into the Manager before we can use it to
    # interact with environment and collect data
    manager.add_agents(agents)
    manager.start()
