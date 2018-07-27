from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
import numpy as np
import random


class Experience(object):
    def __init__(self, quantities):
        assert isinstance(quantities, dict)
        assert "episode_end" in quantities
        self.episode_end = quantities["episode_end"][0]
        self.quantities = quantities

    def is_episode_end(self):
        return self.episode_end

    def val(self, key):
        return self.quantities[key]

    def keys(self):
        return sorted(self.quantities.keys())


class Sample(object):
    """
    A Sample represents one or a sequence of Experiences
    """

    def __init__(self, i, n):
        self.i = i  # starting index of the first experience in the sample
        self.n = n  # length of the sequence

    def __repr__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Create Replay buffer.

        Args:
            exp_type(object): Experience class used in the buffer.
            capacity(int): Max number of experience to store in the buffer. When
                the buffer overflows the old memories are dropped.
        """
        assert capacity > 1
        self.buffer = []  # a circular queue to store experiences
        self.capacity = capacity  # capacity of the buffer
        self.last = -1  # the index of the last element in the buffer

    def __len__(self):
        return len(self.buffer)

    def buffer_end(self, i):
        return i == self.last

    def next_idx(self, i):
        if self.buffer_end(i):
            return -1
        else:
            return (i + 1) % self.capacity

    def add(self, e):
        """
        Store one experience into the buffer.
        """
        assert isinstance(
            e, Experience), "Replay buffer only accepts Experience instances!"
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.last = (self.last + 1) % self.capacity
        self.buffer[self.last] = deepcopy(e)

    def sample(self, num_experiences, num_seqs=0):
        """
        If num_seqs > 0, generate a batch of sequences of experiences.
        Each sequence has a length of int(num_experiences / num_seqs).
        A sequence must not cross the boundary between two games.
        If num_seqs == 0, generate a batch of individual experiences.
        """
        exp_seqs = []
        if len(self.buffer) <= 1000:
            return exp_seqs

        if num_seqs == 0:
            num_seqs = num_experiences
        for _ in xrange(num_seqs):
            while True:
                idx = np.random.randint(0, len(self.buffer) - 1)
                if not self.buffer_end(idx) and not self.buffer[
                        idx].is_episode_end():
                    break
            indices = [idx]
            for i in range(num_experiences / num_seqs):
                if self.buffer_end(idx) or self.buffer[idx].is_episode_end():
                    break
                idx = self.next_idx(idx)
                indices.append(idx)
            exp_seqs.append([deepcopy(self.buffer[idx]) for idx in indices])

        return exp_seqs


class NoReplacementQueue(object):
    def __init__(self):
        self.q = deque()

    def __len__(self):
        return len(self.q)

    def __repr__(self):
        print '[len={0},'.format(len(self))
        for e in self.q:
            print '\t{0},'.format(e)
        print ']'

    def add(self, t):
        self.q.append(deepcopy(t))

    def sample(self):
        exp_seqs = []
        while len(self.q) > 1:
            exps = []
            while len(self.q) > 1 and not self.q[0].is_episode_end():
                # no need to deepcopy here as the selected exp is immediately
                # removed from the queue
                exps.append(self.q.popleft())
            if exps:
                exps.append(deepcopy(self.q[0]))
                exp_seqs.append(exps)
            if self.q[0].is_episode_end():
                self.q.popleft()
        return exp_seqs
