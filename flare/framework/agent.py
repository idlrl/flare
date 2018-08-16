from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Value
import numpy as np
from flare.common.logging import GameLogEntry
from flare.common.communicator import AgentCommunicator
from flare.common.replay_buffer import NoReplacementQueue, ReplayBuffer, Experience


class AgentHelper(object):
    """
    AgentHelper abstracts some part of Agent's data processing and the I/O
    communication between Agent and ComputationDataProcessor (CDP). It receives a
    Communicator from one CDP and uses it to send data to the CDP.
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, communicator, sample_interval):
        assert isinstance(communicator, AgentCommunicator)
        self.name = name
        self.comm = communicator
        self.counter = 0
        self.sample_interval = sample_interval

    def unpack_exps(self, exp_seqs):
        """
        The input `exp_seqs` is always a list of sequences, each sequence
        containing multiple Experience instances.
        """

        def concat_lists(lists):
            return [x for l in lists for x in l]

        def extract_key(seq, k):
            assert seq
            return [e.val(k) for e in seq]

        size = sum([len(exp_seq) - 1 for exp_seq in exp_seqs])
        ret = dict(
            inputs={},
            next_inputs={},
            next_alive={},
            rewards={},
            actions={},
            next_actions={},
            states=None,
            next_states=None)

        for k in self.input_keys:
            ipt_seqs = [extract_key(exp_seq, k) for exp_seq in exp_seqs]
            ret["inputs"][k] = [ipt_seq[:-1] for ipt_seq in ipt_seqs]
            ret["next_inputs"][k] = [ipt_seq[1:] for ipt_seq in ipt_seqs]

        for k in self.action_keys:
            act_seqs = [extract_key(exp_seq, k) for exp_seq in exp_seqs]
            ret["actions"][k] = [act_seq[:-1] for act_seq in act_seqs]
            ret["next_actions"][k] = [act_seq[1:] for act_seq in act_seqs]

        for k in self.reward_keys:
            ret["rewards"][
                k] = [extract_key(exp_seq[:-1], k) for exp_seq in exp_seqs]

        if self.state_keys:
            ret["states"] = dict()
            ret["next_states"] = dict()

        for k in self.state_keys:
            ## we only take the first/second element of a seq
            ret["states"][
                k] = [extract_key(exp_seq[:1], k)[0] for exp_seq in exp_seqs]
            ret["next_states"][k] = [
                extract_key(exp_seq[1:2], k)[0] for exp_seq in exp_seqs
            ]

        ret["next_alive"]["alive"] \
            = [extract_key(exp_seq[1:], "alive") for exp_seq in exp_seqs]

        if not self.state_keys:  # sample instances
            for k in ret.keys():
                if ret[k] is not None:
                    for kk in ret[k].keys():
                        ret[k][kk] = concat_lists(ret[k][kk])

        return ret, size

    def predict(self, inputs, states=dict()):
        """
        Process the input data (if necessary), send them to CDP for prediction,
        and receive the outcome.

        Args:
            inputs(dict): data used for prediction. It is caller's job
            to make sure inputs contains all data needed and they are in the
            right form.
        """
        data = dict(inputs=inputs, states=states)
        self.comm.put_prediction_data((data, 1))
        ret = self.comm.get_prediction_return()
        return ret

    @abstractmethod
    def add_experience(self, e):
        """
        Implements how to record an experience.
        Will be called by self.store_data()
        """
        pass

    def _store_data(self, alive, data):
        """
        Store the past experience for later use, e.g., experience replay.

        Args:
            data(dict): data to store.
        """
        assert isinstance(data, dict)
        data["alive"] = [alive]
        t = Experience(data)
        self.add_experience(t)
        self.counter += 1
        if self.counter % self.sample_interval == 0:
            self.learn()

    @abstractmethod
    def sample_experiences(self):
        """
        Implements how to retrieve experiences from past.
        Will be called by self.learn()
        """
        pass

    def learn(self):
        """
        Sample data from past experiences and send them to CDP for learning.
        Optionally, it receives learning outcomes sent back from CW and does
        some processing.

        Depends on users' need, this function can be called in three ways:
        1. In Agent's run_one_episode
        2. In store_data(), e.g., learning once every few steps
        3. As a separate thread, e.g., using experience replay
        """
        exp_seqs = self.sample_experiences()
        if not exp_seqs:
            return
        data, size = self.unpack_exps(exp_seqs)
        self.comm.put_training_data((data, size))
        ret = self.comm.get_training_return()


class OnPolicyHelper(AgentHelper):
    """
    On-policy helper. It calls `learn()` every `sample_interval`
    steps.

    While waiting for learning return, the calling `Agent` is blocked.
    """

    def __init__(self, name, communicator, sample_interval=5):
        super(OnPolicyHelper, self).__init__(name, communicator,
                                             sample_interval)
        # NoReplacementQueue used to store past experience.
        self.exp_queue = NoReplacementQueue()

    @staticmethod
    def exp_replay():
        return False

    def add_experience(self, e):
        self.exp_queue.add(e)

    def sample_experiences(self):
        return self.exp_queue.sample()


class ExpReplayHelper(AgentHelper):
    """
    Example of applying experience replay. It starts a separate threads to
    run learn().
    """

    def __init__(self,
                 name,
                 communicator,
                 buffer_capacity,
                 num_experiences,
                 sample_interval=5,
                 num_seqs=1):
        super(ExpReplayHelper, self).__init__(name, communicator,
                                              sample_interval)
        # replay buffer for experience replay
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.num_experiences = num_experiences
        self.num_seqs = num_seqs

    @staticmethod
    def exp_replay():
        return True

    def add_experience(self, e):
        self.replay_buffer.add(e)

    def sample_experiences(self):
        return self.replay_buffer.sample(self.num_experiences, self.num_seqs)


class Agent(Process):
    """
    Agent implements the control flow and logics of how Robot interacts with
    the environment and does computation. It is a subclass of Process. The entry
    function of the Agent process is run().

    Agent has the following members:
    env: the environment
    num_games:  number of games to run
    helpers:    a dictionary of `AgentHelper`, each corresponds to one
                `ComputationTask`
    log_q:      communication channel between `Agent` and the centralized logger
    running:    the `Agent` will keep running as long as `running` is True.
    """
    __metaclass__ = ABCMeta

    def __init__(self, env, num_games):
        super(Agent, self).__init__()
        self.id = -1  # just created, not added to the Robot yet
        self.env = env
        self.num_games = num_games
        self.state_specs = None
        self.helpers = {}
        self.log_q = None
        self.running = Value('i', 0)
        self.daemon = True
        self.alive = 1

    def add_agent_helper(self, helper, input_keys, action_keys, state_keys,
                         reward_keys):
        """
        Add an AgentHelper, with its name (also the name of its
        correspoding `ComputationTask`) as key.
        """
        assert isinstance(helper, AgentHelper)
        helper.input_keys = input_keys
        helper.action_keys = action_keys
        helper.state_keys = state_keys
        helper.reward_keys = reward_keys
        self.helpers[helper.name] = helper

    def make_initial_states(self, specs_dict):
        self.init_states = {}
        for alg_name, specs_list in specs_dict.iteritems():
            states = {}
            for specs in specs_list:
                dtype = specs[1]["dtype"] if "dtype" in specs[1] else "float32"
                states[specs[0]] = np.zeros(specs[1]["shape"]).astype(dtype)
            self.init_states[alg_name] = states

    ## The following three functions hide the `AgentHelper` from the users of
    ## `Agent`.
    def predict(self, alg_name, inputs, states=dict()):
        return self.helpers[alg_name].predict(inputs, states)

    def learn(self, alg_name):
        self.helpers[alg_name].learn()

    def run(self):
        """
        Default entry function of Agent process.
        """
        self.running.value = 1
        for i in range(self.num_games):
            self._run_one_episode()
            if not self.running.value:
                return
        self.running.value = 0

    def _store_data(self, alg_name, data):
        self.helpers[alg_name]._store_data(self.alive, data)

    def _run_one_episode(self):
        observations = self._reset_env()
        states = self._get_init_states()  ## written by user

        while self.alive and (not self._game_timeout()):
            actions, next_states = self._cts_predict(
                observations, states)  ## written by user
            assert isinstance(actions, list)
            assert isinstance(next_states, list)
            next_observations, rewards, next_game_over = self._step_env(
                actions)
            self._cts_store_data(observations, actions, states,
                                 rewards)  ## written by user
            observations = next_observations
            states = next_states
            self.alive = 1 - int(next_game_over)

        actions, _ = self._cts_predict(observations, states)
        if self._game_timeout():
            self.alive = -1
        self._cts_store_data(observations, actions, states, [0] * len(rewards))

        return self._total_reward()

    def _reset_env(self):
        self.alive = 1
        self.steps = 0
        ## currently we only support a single logger for all CTs
        self.log_entry = GameLogEntry(self.id, 'All')
        obs = self.env.reset()
        assert isinstance(obs, list)
        self.max_steps = self.env.get_max_steps()
        return obs

    def _game_timeout(self):
        ## For OpenAI gym, end one step earlier to avoid
        ## getting the game_over signal

        return self.steps >= self.max_steps - 1

    def _step_env(self, actions):
        self.steps += 1
        next_observations, rewards, next_game_over = self.env.step(actions)
        assert isinstance(next_observations, list)
        assert isinstance(rewards, list)
        self.log_entry.num_steps += 1
        self.log_entry.total_reward += sum(rewards)
        return next_observations, rewards, next_game_over

    def _total_reward(self):
        self.log_q.put(self.log_entry)
        return self.log_entry.total_reward

    def _get_init_states(self):
        """
        By default, there is no state. The user needs to override this function
        to return a list of init states if necessary.
        """
        return []

    @abstractmethod
    def _cts_predict(self, observations, states):
        """
        The user needs to override this function to specify how different CTs
        make predictions given observations and states.

        Output: actions:           a list of actions, each action being a vector
                                   If the action is discrete, then it is a length-one
                                   list of an integer.
                states (optional): a list of states, each state being a floating vector
        """
        pass

    @abstractmethod
    def _cts_store_data(self, observations, actions, states, rewards):
        """
        The user needs to override this function to specify how different CTs
        store their corresponding experiences, by calling self._store_data().
        """
        pass
