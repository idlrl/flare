from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Value
import numpy as np
from flare.common.log import GameLogEntry
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
        assert sample_interval >= 2
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
            ## we only take the first(second) element of a seq for states(next_states)
            ret["states"][
                k] = [extract_key(exp_seq[:1], k)[0] for exp_seq in exp_seqs]
            ret["next_states"][k] = [
                extract_key(exp_seq[1:2], k)[0] for exp_seq in exp_seqs
            ]

        ret["next_alive"]["alive"] \
            = [extract_key(exp_seq[1:], "alive") for exp_seq in exp_seqs]

        ## HERE we decide whether the data are instances or seqs
        ## according to the existence of states
        if not self.state_keys:
            # sample instances
            for k in ret.keys():
                if ret[k] is not None:
                    for kk in ret[k].keys():
                        ret[k][kk] = concat_lists(ret[k][kk])

        return ret, len(exp_seqs)

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
        self.comm.put_prediction_data(data, 1)
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
            return self.learn()

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
        self.comm.put_training_data(data, size)
        ret = self.comm.get_training_return()
        return ret


class OnlineHelper(AgentHelper):
    """
    Online helper. It calls `learn()` every `sample_interval`
    steps.

    While waiting for learning return, the calling `Agent` is blocked.
    """

    def __init__(self, name, communicator, sample_interval=5):
        super(OnlineHelper, self).__init__(name, communicator, sample_interval)
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

    Some members:
    env:        the environment
    num_games:  number of games to run
    learning:   Whether learn or not (only do testing)
    helpers:    a dictionary of `AgentHelper`, each corresponds to one
                `ComputationTask`
    log_q:      communication channel between `Agent` and the centralized logger
    running:    the `Agent` will keep running as long as `running` is True.
    """
    __metaclass__ = ABCMeta

    def __init__(self, num_games, actrep, learning):
        super(Agent, self).__init__()
        self.id = -1  # just created, not added to the Robot yet
        self.num_games = num_games
        self.learning = learning
        self.state_specs = None
        self.helpers = {}
        self.log_q = None
        self.running = Value('i', 0)
        self.daemon = True  ## Process member
        self.alive = 1
        self.env_f = None
        self.actrep = actrep

    def set_env(self, env_class, *args, **kwargs):
        """
        Set the environment for the agent. For now, only create a lambda
        function. Once the agent process starts running, we will call this
        function.

        env_class:    The environment class to create
        args, kwargs: The arguments for creating the class
        """
        self.env_f = lambda: env_class(*args, **kwargs)

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

    def _make_zero_states(self, prop):
        dtype = prop["dtype"] if "dtype" in prop else "float32"
        return np.zeros(prop["shape"]).astype(dtype)

    ## The following three functions hide the `AgentHelper` from the users of
    ## `Agent`.
    def predict(self, alg_name, inputs, states=dict()):
        ## Convert single instances to batches of size 1
        ## The reason for this conversion is that we want to reuse the
        ## _pack_data() and _unpack_data() of the CDP for handling both training
        ## and prediction data. These two functions assume that data are stored
        ## as mini batches instead of single instances in the prediction and learning
        ## queues.
        inputs_ = {k: [v] for k, v in inputs.items()}
        states_ = {k: [v] for k, v in states.items()}
        prediction, next_states = self.helpers[alg_name].predict(inputs_,
                                                                 states_)
        ## convert back to single instances
        prediction = {k: v[0] for k, v in prediction.items()}
        next_states = {k: v[0] for k, v in next_states.items()}
        return prediction, next_states

    def run(self):
        """
        Default entry function of Agent process.
        """
        assert self.env_f is not None, "You should first call self.set_env()!"
        ## Only call the env function now to make sure there is only one
        ## environment (OpenGL context) in each process
        self.env = self.env_f()
        self.running.value = 1
        for i in range(self.num_games):
            self._run_one_episode()
            if not self.running.value:
                return
        self.running.value = 0

    def _store_data(self, alg_name, data):
        if self.learning:  ## only store when the agent is learning
            return self.helpers[alg_name]._store_data(self.alive, data)

    def _run_one_episode(self):
        def __store_data(observations, actions, states, rewards):
            learning_ret = self._cts_store_data(observations, actions, states,
                                                rewards)  ## written by user
            if learning_ret is not None:
                for k, v in learning_ret.items():
                    self.log_entry.add_key(k, v)

        observations = self._reset_env()
        states = self._get_init_states()  ## written by user

        while self.alive and (not self.env.time_out()):
            actions, next_states = self._cts_predict(
                observations, states)  ## written by user
            assert isinstance(actions, dict)
            assert isinstance(next_states, dict)
            next_observations, rewards, next_game_over = self._step_env(
                actions)
            __store_data(observations, actions, states, rewards)

            observations = next_observations
            states = next_states
            ## next_game_over == 1:  success
            ## next_game_over == -1: failure
            self.alive = 1 - abs(next_game_over)

        ## self.alive:  0  -- success/failure
        ##              1  -- normal
        ##             -1  -- timeout
        if self.env.time_out():
            self.alive = -1
        actions, _ = self._cts_predict(observations, states)
        zero_rewards = {k: [0] * len(v) for k, v in rewards.items()}
        __store_data(observations, actions, states, zero_rewards)

        ## Record success. For games that do not have a defintion of
        ## 'success' (e.g., 'breakout' never ends), this quantity will
        ## always be zero
        self.log_entry.add_key("success", next_game_over > 0)
        return self._total_reward()

    def _reset_env(self):
        self.alive = 1
        ## currently we only support a single logger for all CTs
        self.log_entry = GameLogEntry(self.id, 'All')
        obs = self.env.reset()
        assert isinstance(obs, dict)
        return obs

    def _step_env(self, actions):
        next_observations, rewards, next_game_over = self.env.step(actions,
                                                                   self.actrep)
        assert isinstance(next_observations, dict)
        assert isinstance(rewards, dict)
        self.log_entry.add_key("num_steps", 1)
        self.log_entry.add_key("total_reward", sum(map(sum, rewards.values())))
        return next_observations, rewards, next_game_over

    def _total_reward(self):
        self.log_q.put(self.log_entry)
        return self.log_entry.total_reward

    def _get_init_states(self):
        """
        By default, there is no state. The user needs to override this function
        to return a dictionary of init states if necessary.
        """
        return dict()

    @abstractmethod
    def _cts_predict(self, observations, states):
        """
        The user needs to override this function to specify how different CTs
        make predictions given observations and states.

        Output: actions:           a dictionary of actions, each action being a vector
                                   If the action is discrete, then it is a length-one
                                   list of an integer.
                states (optional): a dictionary of states, each state being a floating vector
        """
        pass

    @abstractmethod
    def _cts_store_data(self, observations, actions, states, rewards):
        """
        The user needs to override this function to specify how different CTs
        store their corresponding experiences, by calling self._store_data().
        Each input should be a dictionary.
        """
        pass
