import glog
from collections import deque
from multiprocessing import Queue, Process, Value
from Queue import Empty, Full


class Statistics(object):
    def __init__(self, keys, moving_window=0):
        self.keys = keys
        self.total = {k: 0 for k in keys}
        self.data_q = {k: deque(maxlen=moving_window) for k in keys}
        self.num_games = 0

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
        for k in self.keys:
            v = getattr(log, k)
            self.total[k] += v
            self.data_q[k].append(v)
        self.num_games += 1

    def record_logs(self, logs):
        for k in self.keys:
            D = [getattr(l, k) for l in logs]
            self.total[k] = sum(D)
            self.data_q[k].extend(D)
        self.num_games += len(D)


class GameLogEntry(object):
    """
    GameLogEntry records the statistics of one game.
    """

    def __init__(self, agent_id, alg_name, num_steps=0, total_reward=0):
        self.agent_id = agent_id
        self.alg_name = alg_name
        self.num_steps = num_steps
        self.total_reward = total_reward

    @classmethod
    def get_stats(cls, moving_window):
        return Statistics(["num_steps", "total_reward"], moving_window)


class GameLogger(Process):
    def __init__(self, timeout, print_interval, model_save_interval):
        super(GameLogger, self).__init__()
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
            glog.info('\n{0}:{1}'.format(alg_name, stats))

    def __save_models(self, idx):
        for signal in self.model_save_signals:
            signal.value = idx

    def __process_log(self, log):
        if not log.alg_name in self.stats:
            self.stats[log.alg_name] = log.get_stats(self.print_interval)
        self.stats[log.alg_name].record_one_log(log)
        self.counter += 1
        if self.counter % self.print_interval == 0:
            self.__flush_log()
            if self.counter % (self.print_interval * self.model_save_interval
                               ) == 0:
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
