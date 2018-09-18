import numpy as np
from flare.framework.agent import Agent


class SimpleRLAgent(Agent):
    """
    This class serves as an example of simple RL algorithms, which has only one
    ComputationTask, "RL", i.e., using and learning an RL policy.

    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self,
                 env,
                 num_games,
                 learning=True,
                 reward_shaping_f=lambda x: x):
        super(SimpleRLAgent, self).__init__(env, num_games, learning)
        self.reward_shaping_f = reward_shaping_f

    def _cts_store_data(self, observations, actions, states, rewards):
        assert len(observations) == 1 and len(actions) == 1
        self._store_data(
            'RL',
            dict(
                sensor=observations[0],
                action=actions[0],
                reward=[self.reward_shaping_f(r) for r in rewards]))

    def _cts_predict(self, observations, states):
        assert len(observations) == 1
        actions, _ = self.predict('RL', inputs=dict(sensor=observations[0]))
        return [actions["action"]], []


class SimpleRNNRLAgent(Agent):
    """
    This class serves as an example of simple RL algorithms with a single RNN state,
    which has only one ComputationTask, "RL", i.e., using and learning an RL policy.

    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self,
                 env,
                 num_games,
                 learning=True,
                 reward_shaping_f=lambda x: x):
        super(SimpleRNNRLAgent, self).__init__(env, num_games, learning)
        self.reward_shaping_f = reward_shaping_f

    def _get_init_states(self):
        return [
            self._make_zero_states(prop)
            for _, prop in self.cts_state_specs['RL']
        ]

    def _cts_store_data(self, observations, actions, states, rewards):
        assert len(observations) == 1 and len(actions) == 1
        assert len(states) == 1
        self._store_data(
            'RL',
            dict(
                sensor=observations[0],
                action=actions[0],
                state=states[0],
                reward=[self.reward_shaping_f(r) for r in rewards]))

    def _cts_predict(self, observations, states):
        assert len(observations) == 1 and len(states) == 1
        actions, next_states = self.predict(
            'RL',
            inputs=dict(sensor=observations[0]),
            states=dict(state=states[0]))
        return [actions["action"]], next_states.values()


class ActionNoiseAgent(SimpleRLAgent):
    """
    This class extends `SimpleRLAgent` by applying action noise after 
    prediction. It can be used to algorithms with deterministic policies, e.g.,
    `DDPG`.
    """

    def __init__(self,
                 env,
                 num_games,
                 action_noise,
                 reward_shaping_f=lambda x: x):
        super(ActionNoiseAgent, self).__init__(env, num_games,
                                               reward_shaping_f)
        self.action_noise = action_noise

    def _cts_predict(self, observations, states):
        ## each action is already 2D
        assert len(observations) == 1
        actions, _ = self.predict('RL', inputs=dict(sensor=observations[0]))
        a = actions.values()[0][0]
        a = a + self.action_noise.noise()

        return [a], []

    def _reset_env(self):
        self.action_noise.reset()
        return super(ActionNoiseAgent, self)._reset_env()
