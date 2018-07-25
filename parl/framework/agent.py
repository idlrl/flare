from abc import ABCMeta, abstractmethod
from multiprocessing import Process, Value
import numpy as np
from threading import Lock, Thread
from parl.common.communicator import AgentCommunicator
from parl.common.replay_buffer import NoReplacementQueue, ReplayBuffer


class AgentHelper(object):
    """
    AgentHelper abstracts some part of Agent's data processing and the I/O 
    communication between Agent and ComputationDataProcessor (CP). It receives a
    Communicator from one CDP and uses it to send data to the CDP.
    """
    __metaclass__ = ABCMeta

    __on_policy = False

    def __init__(self, name, on_policy, communicator):
        assert isinstance(communicator, AgentCommunicator)
        self.name = name
        self.__on_policy = on_policy
        self.comm = communicator
        self.pack_func = None
        self.unpack_func = None
        self.is_episode_end = None

    def start(self):
        pass

    def stop(self):
        pass

    @classmethod
    def on_policy(cls):
        return self.__on_policy

    @abstractmethod
    def predict(self, inputs, states):
        """
        Process the input data (if necessary), send them to CDP for prediction,
        and receive the outcome.

        Args:
            inputs(dict): data used for prediction. It is caller's job 
            to make sure inputs contains all data needed and they are in the 
            right form.
        """
        pass

    def store_data(self, **kwargs):
        """
        Store the past experience for later use, e.g., experience replay.

        Args:
            data(dict): data to store. 
        """
        pass

    @abstractmethod
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
        pass


class OnPolicyHelper(AgentHelper):
    """
    On-policy helper. It calls `learn()` every `sample_interval`
    steps.

    While waiting for learning return, the calling `Agent` is blocked.
    """

    def __init__(self, name, communicator, sample_interval=5,
                 sample_seq=False):
        super(OnPolicyHelper, self).__init__(name, True, communicator)
        self.sample_interval = sample_interval
        # NoReplacementQueue used to store past experience.
        # TODO: support sequence sampling 
        self.exp_queue = NoReplacementQueue()
        self.counter = 0

    def predict(self, inputs, states=dict()):
        data = dict(inputs=inputs, states=states)
        self.comm.put_prediction_data((data, 1))
        ret = self.comm.get_prediction_return()
        return ret

    def store_data(self, **kwargs):
        t = self.pack_func(**kwargs)
        self.exp_queue.add(t)
        self.counter += 1
        if self.counter % self.sample_interval == 0:
            self.learn()
            self.counter = 0

    def learn(self):
        exp_seqs = self.exp_queue.sample(self.is_episode_end)
        data, size = self.unpack_func(exp_seqs)
        self.comm.put_training_data((data, size))
        self.comm.get_training_return()


class ExpReplayHelper(AgentHelper):
    """
    Example of applying experience replay. It starts a separate threads to
    run learn().
    """

    def __init__(self, name, communicator, buffer_capacity, sample_size,
                 num_seqs):
        super(ExpReplayHelper, self).__init__(name, False, communicator)
        # replay buffer for experience replay
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.sample_size = sample_size
        self.num_seqs = num_seqs
        # the thread that will run learn()
        self.learning_thread = Thread(target=self.learn)
        # prevent race on the replay_buffer
        self.lock = Lock()
        # flag to signal learning_thread to stop
        self.running = Value('i', 0)

    def start(self):
        self.running.value = 1
        self.learning_thread.start()

    def stop(self):
        self.running.value = 0
        self.learning_thread.join()

    def predict(self, inputs, states=dict()):
        data = dict(inputs=inputs, states=states)
        self.comm.put_prediction_data((data, 1))
        ret = self.comm.get_prediction_return()
        return ret

    def store_data(self, **kwargs):
        t = self.pack_func(**kwargs)
        with self.lock:
            self.replay_buffer.add(t)

    def learn(self):
        """
        This function should be invoked in a separate thread. Once called, it
        keeps sampling data from the replay buffer until exit_flag is signaled.
        """
        # keep running until exit_flag is signaled
        while self.running.value:
            with self.lock:
                exp_seqs = self.replay_buffer.sample(
                    self.sample_size, self.is_episode_end, self.num_seqs)
            if not exp_seqs:
                continue
            data, size = self.unpack_func(exp_seqs)
            self.comm.put_training_data((data, size))
            ret = self.comm.get_training_return()


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
    running:    the `Agetn` will keep running as long as `running` is True.
    """
    __metaclass__ = ABCMeta

    def __init__(self, env, num_games):
        super(Agent, self).__init__()
        self.id = -1  # just created, not added to the Robot yet
        self.env = env
        self.num_games = num_games
        self.state_specs = None
        self.helpers = {}
        self.log_q = []
        self.running = Value('i', 0)
        self.daemon = True

    def add_agent_helper(self, helper, pack_f, unpack_f, is_episode_end_f):
        """
        Add an AgentHelper, with its name (also the name of its
        correspoding `ComputationTask`) as key.
        """
        assert isinstance(helper, AgentHelper)
        assert callable(pack_f)
        assert callable(unpack_f)
        assert callable(is_episode_end_f)
        helper.pack_func = pack_f
        helper.unpack_func = unpack_f
        helper.is_episode_end = is_episode_end_f
        self.helpers[helper.name] = helper

    def make_initial_states(self, specs_dict):
        self.init_states = {}
        for alg_name, specs_list in specs_dict.iteritems():
            states = {}
            for specs in specs_list:
                dtype = specs[1]["dtype"] if "dtype" in specs[1] else "float32"
                states[specs[0]] = np.zeros([1] + specs[1]["shape"]).astype(
                    dtype)
            self.init_states[alg_name] = states
        return self.init_states

    @abstractmethod
    def pack_exps(cls, **kwargs):
        """
        Process the experience data before storing them. This function will be
        given to `AgentHelper` as its attribute and called from `store_data`.

        User should implement this class method to accommodate their needs.
        """
        pass

    @abstractmethod
    def unpack_exp_seqs(cls, **kwargs):
        """
        Process the experience data before sending them for learning. This 
        function will be given to `AgentHelper` as its attribute and called from
        `learn`.

        User should implement this class method to accommodate their needs.
        """

        pass

    @abstractmethod
    def is_episode_end(cls, t):
        """
        Given an experience, return True if it represents episode end.

        User should implement this class method based on the content of `t`.

        Args:
            t: experience data of one time step. It can be any type.
        """
        pass

    @abstractmethod
    def _run_one_episode(self):
        """
        This function implements the control flow of running one episode, which
        includes:
        1. The interaction with the environment
        2. Calls AgentHelper's interfaces to process the data 
        """
        pass

    ## The following three functions hide the `AgentHelper` from the users of
    ## `Agent`.
    def predict(self, alg_name, inputs, states=dict()):
        return self.helpers[alg_name].predict(inputs, states)

    def store_data(self, alg_name, **kwargs):
        self.helpers[alg_name].store_data(**kwargs)

    def learn(self, alg_name):
        self.helpers[alg_name].learn()

    def run(self):
        """
        Entry function of Agent process.
        """
        self.running.value = 1
        for helper in self.helpers.itervalues():
            helper.start()
        for i in range(self.num_games):
            self._run_one_episode()
            if not self.running.value:
                return
        self.running.value = 0
        for helper in self.helpers.itervalues():
            helper.stop()
