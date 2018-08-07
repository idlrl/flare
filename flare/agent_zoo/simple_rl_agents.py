import numpy as np
from flare.framework.agent import Agent


class SimpleRLAgent(Agent):
    """
    This class serves as an example of simple RL algorithms, which has only one
    ComputationTask, "RL", i.e., using and learning an RL policy.

    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self, env, num_games):
        super(SimpleRLAgent, self).__init__(env, num_games)

    def _cts_store_data(self, observations, actions, states, rewards):
        assert len(observations) == 1 and len(actions) == 1
        self._store_data(
            'RL',
            dict(
                sensor=observations[0],
                action=actions[0],
                reward=[r / 100.0 for r in rewards]))

    def _cts_predict(self, observations, states):
        ## each action is already 2D
        assert len(observations) == 1
        actions, _ = self.predict('RL', inputs=dict(sensor=observations))
        return [actions.values()[0][0]], []


class SimpleRNNRLAgent(Agent):
    """
    This class serves as an example of simple RL algorithms with a single RNN state,
    which has only one ComputationTask, "RL", i.e., using and learning an RL policy.

    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self, env, num_games):
        super(SimpleRNNRLAgent, self).__init__(env, num_games)

    def _get_init_states(self):
        return self.init_states['RL'].values()

    def _cts_store_data(self, observations, actions, states, rewards):
        assert len(observations) == 1 and len(actions) == 1
        assert len(states) == 1
        self._store_data(
            'RL',
            dict(
                sensor=observations[0],
                action=actions[0],
                state=states[0],
                reward=[r / 100.0 for r in rewards]))

    def _cts_predict(self, observations, states):
        ## each action is already 2D
        assert len(observations) == 1 and len(states) == 1
        actions, next_states = self.predict(
            'RL', inputs=dict(sensor=observations), states=dict(state=states))
        return [actions.values()[0][0]], [next_states.values()[0][0]]
