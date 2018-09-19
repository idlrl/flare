import numpy as np
from flare.framework.agent import Agent
from flare.framework import common_functions as comf


class SimpleRLAgent(Agent):
    """
    This class serves as an example of simple RL algorithms, which has only one
    ComputationTask, "RL", i.e., using and learning an RL policy.

    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self,
                 num_games,
                 actrep=1,
                 learning=True,
                 reward_shaping_f=lambda x: x):
        super(SimpleRLAgent, self).__init__(num_games, actrep, learning)
        self.reward_shaping_f = reward_shaping_f

    def _cts_store_data(self, observations, actions, states, rewards):
        ## before storing rewards for training, we reshape them
        for k in rewards.keys():
            ## each r is a reward vector
            rewards[k] = [self.reward_shaping_f(r) for r in rewards[k]]
        ## store everything in the buffer
        data = {}
        data.update(observations)
        data.update(actions)
        data.update(states)
        data.update(rewards)
        ret = self._store_data('RL', data)
        if ret is not None:
            ## If not None, _store_data calls learn() in this iteration
            ## We return the cost for logging
            cost, learn_info = ret
            return {k: comf.sum_cost_array(v)[0] for k, v in cost.iteritems()}

    def _cts_predict(self, observations, states):
        return self.predict('RL', observations, states)


class SimpleRNNRLAgent(SimpleRLAgent):
    """
    This class serves as an example of simple RL algorithms with a single RNN state,
    which has only one ComputationTask, "RL", i.e., using and learning an RL policy.

    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self,
                 num_games,
                 actrep=1,
                 learning=True,
                 reward_shaping_f=lambda x: x):
        super(SimpleRNNRLAgent, self).__init__(num_games, actrep, learning,
                                               reward_shaping_f)

    def _get_init_states(self):
        return {name : self._make_zero_states(prop) \
                for name, prop in self.cts_state_specs['RL']}
