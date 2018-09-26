import logging
import sys
from collections import deque
from multiprocessing import Queue, Process, Value
from Queue import Empty, Full


class Statistics(object):
    def __init__(self, moving_window=0):
        self.keys = []
        self.total = {}
        self.data_q = {}
        self.num_games = 0
        self.moving_window = moving_window

    def __repr__(self):
        str = '[\n    num_games={0}\n'.format(self.num_games)
        for k in self.keys:
            str += '    {0} [total: {1}, average@{2}: {3}]\n'.format(
                k, self.total[k],
                len(self.data_q[k]),
                sum(self.data_q[k]) / float(len(self.data_q[k])))
        str += ']'
        return str

    def record_one_log(self, log):
        for k in log.log_keys:
            if k not in self.keys:
                self.keys.append(k)
                self.total[k] = 0
                self.data_q[k] = deque(maxlen=self.moving_window)
            v = getattr(log, k)
            self.total[k] += v
            self.data_q[k].append(v)
        self.num_games += 1


class GameLogEntry(object):
    """
    GameLogEntry records the statistics of one game.
    """

    def __init__(self, agent_id, alg_name):
        self.agent_id = agent_id
        self.alg_name = alg_name
        self.log_keys = []

    def add_key(self, key, value):
        """
        All quantities added by this function will be reported
        as accumulated values across a game on average
        """
        try:
            a = getattr(self, key)
        except:
            a = 0
            self.log_keys.append(key)
        setattr(self, key, value + a)


class GameLogger(Process):
    def __init__(self, timeout, print_interval, model_save_interval, log_file):
        super(GameLogger, self).__init__()
        kwargs = dict(
            format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
            datefmt='%d-%m-%Y:%H:%M:%S',
            level=logging.DEBUG)
        if log_file != "":
            kwargs["filename"] = log_file
        else:
            kwargs["stream"] = sys.stdout

        logging.basicConfig(**kwargs)

        self.timeout = timeout
        self.print_interval = print_interval
        self.model_save_interval = model_save_interval
        self.model_save_signals = []
        self.stats = {}
        self.running = Value('i', 0)
        self.log_q = Queue()
        self.counter = 0
        self.daemon = True

    def __flush_log(self):
        for alg_name, stats in self.stats.iteritems():
            logging.info('\n{0}:{1}'.format(alg_name, stats))

    def __save_models(self, idx):
        ## When agent.learning=False, this will not save models
        ## because the CTs are blocked by the learning queues
        for signal in self.model_save_signals:
            signal.value = idx

    def __process_log(self, log):
        if not log.alg_name in self.stats:
            self.stats[log.alg_name] = Statistics(self.print_interval)
        self.stats[log.alg_name].record_one_log(log)
        self.counter += 1
        if self.counter % self.print_interval == 0:
            self.__flush_log()
            if self.counter % (self.print_interval * self.model_save_interval
                               ) == 0:
                ## No matter which pass is loaded, the model will be saved starting
                ## from pass 1
                self.__save_models(self.counter / self.print_interval /
                                   self.model_save_interval)

    def run(self):
        self.running.value = True
        while self.running.value:
            try:
                log = self.log_q.get(timeout=self.timeout)
            except Empty:
                continue
            self.__process_log(log)
        self.__flush_log()
