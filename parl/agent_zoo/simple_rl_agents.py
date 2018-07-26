import numpy as np
from threading import Thread, Lock
from parl.common.logging import GameLogEntry
from parl.framework.agent import Agent


class SimpleRLAgent(Agent):
    """
    This class serves as a template of simple RL algorithms, which has only one
    ComputationTask, "RL", i.e., using and learning an RL policy. 
    
    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self, env, num_games):
        super(SimpleRLAgent, self).__init__(env, num_games)
        self.log_q = None

    @classmethod
    def pack_exps(cls, **kwargs):
        keys = ["sensor", "action", "reward", "episode_end"]
        t = tuple(kwargs[k] for k in keys)
        return t

    @classmethod
    def unpack_exp_seqs(cls, exp_seqs):
        size = sum([len(exp_seq) - 1 for exp_seq in exp_seqs])

        t = zip(*[[np.array(l) for l in zip(*exp_seq[:-1])] \
                  for exp_seq in exp_seqs])
        t_next = zip(*[[np.array(l) for l in zip(*exp_seq[1:])] \
                  for exp_seq in exp_seqs])
        sensor, action, reward = [np.concatenate(t[i]) for i in [0, 1, 2]]
        next_sensor, next_episode_end = [
            np.concatenate(t_next[i]) for i in [0, 3]
        ]
        data = dict(
            inputs=dict(sensor=sensor),
            next_inputs=dict(sensor=next_sensor),
            actions=dict(action=action),
            rewards=dict(reward=reward),
            next_episode_end=dict(next_episode_end=next_episode_end))
        return data, size

    @classmethod
    def is_episode_end(cls, t):
        return t[3][0]

    def _run_one_episode(self):
        # sensor_inputs, (prev_)states and actions are all dict
        max_steps = self.env._max_episode_steps
        obs = self.env.reset()
        episode_end = False
        r = 0
        log_entry = GameLogEntry(self.id, 'RL')
        # end before the Gym wrongly gives game_over=True for a timeout case
        for t in range(max_steps - 1):
            #self.env.render()
            actions, _ = self.predict(
                'RL', inputs=dict(sensor=np.array([obs]).astype("float32")))
            a = actions["action"][0]
            next_obs, r, next_episode_end, _ = self.env.step(a[0])
            r /= 100.0

            log_entry.num_steps += 1
            log_entry.total_reward += r
            self.store_data(
                'RL',
                sensor=obs,
                action=a,
                reward=[r],
                episode_end=[episode_end])
            obs = next_obs
            episode_end = next_episode_end
            if episode_end:
                break
        # we call `predict` one more time to get actions. Needed in case of
        # non-episode-end ending.
        actions, _ = self.predict(
            'RL', inputs=dict(sensor=np.array([obs]).astype("float32")))
        self.store_data(
            'RL',
            sensor=obs,
            action=actions["action"][0],
            reward=[0],
            episode_end=[episode_end])
        self.log_q.put(log_entry)
        return log_entry.total_reward


class SimpleRNNRLAgent(Agent):
    """
    This class serves as a template of simple RL algorithms, which has only one
    ComputationTask, "RL", i.e., using and learning an RL policy. 
    
    By using different AgentHelpers, this Agent can be applied to either on-
    policy or off-policy RL algorithms.
    """

    def __init__(self, env, num_games):
        super(SimpleRNNRLAgent, self).__init__(env, num_games)
        self.log_q = None

    @classmethod
    def pack_exps(cls, **kwargs):
        keys = ["sensor", "state", "action", "reward", "episode_end"]
        t = tuple(kwargs[k] for k in keys)
        return t

    @classmethod
    def unpack_exp_seqs(cls, exp_seqs):
        size = len(exp_seqs)
        t = [[np.array(l) for l in zip(*exp_seq[:-1])] for exp_seq in exp_seqs]
        sensor = [l[0] for l in t]
        state = np.concatenate([l[1][0] for l in t])
        action = [l[2] for l in t]
        reward = [l[3] for l in t]

        t_next = [[np.array(l) for l in zip(*exp_seq[1:])]
                  for exp_seq in exp_seqs]
        next_sensor = [l[0] for l in t_next]
        next_state = np.concatenate([l[1][0] for l in t_next])
        next_episode_end = [l[4] for l in t_next]
        data = dict(
            inputs=dict(sensor=sensor),
            next_inputs=dict(sensor=next_sensor),
            states=dict(state=state),
            next_states=dict(state=next_state),
            actions=dict(action=action),
            rewards=dict(reward=reward),
            next_episode_end=dict(next_episode_end=next_episode_end))
        return data, size

    @classmethod
    def is_episode_end(cls, t):
        return t[4][0]

    def _run_one_episode(self):
        # sensor_inputs, (prev_)states and actions are all dict
        max_steps = self.env._max_episode_steps
        obs = self.env.reset()
        episode_end = False
        r = 0
        log_entry = GameLogEntry(self.id, 'RL')
        state = self.init_states['RL']["state"]
        # end before the Gym wrongly gives game_over=True for a timeout case
        for t in range(max_steps - 1):
            actions, next_states = self.predict(
                'RL',
                inputs=dict(sensor=np.array([obs]).astype("float32")),
                states=dict(state=state))
            a = actions["action"][0][0]
            next_obs, r, next_episode_end, _ = self.env.step(a)
            r /= 100.0

            log_entry.num_steps += 1
            log_entry.total_reward += r
            self.store_data(
                'RL',
                sensor=obs,
                state=state,
                action=[a],
                reward=[r],
                episode_end=[episode_end])
            obs = next_obs
            episode_end = next_episode_end
            state = next_states["state"]
            if episode_end:
                break
        # we call `predict` one more time to get actions. Needed in case of
        # non-episode-end ending.
        actions, next_states = self.predict(
            'RL',
            inputs=dict(sensor=np.array([obs]).astype("float32")),
            states=dict(state=state))
        self.store_data(
            'RL',
            sensor=obs,
            state=state,
            action=actions["action"][0],
            reward=[0],
            episode_end=[episode_end])
        self.log_q.put(log_entry)
        return log_entry.total_reward
