#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from collections import deque
import numpy as np
import random


class Experience(object):
    def set_next_exp(self, next_exp):
        self.next_exp = deepcopy(next_exp)

    @staticmethod
    def define(name, attrs):
        """
        Create an Experience 
        """

        def set_attributes(self, **kwargs):
            for k, v in kwargs.iteritems():
                if not hasattr(self, k):
                    raise TypeError
                setattr(self, k, v)

        assert isinstance(attrs, list)
        cls_attrs = dict((attr, None) for attr in attrs)
        cls_attrs['next_exp'] = None  # add attribute "next_exp"
        # __init__ of the new Experience class
        cls_attrs['__init__'] = set_attributes
        cls = type(name, (Experience, ), cls_attrs)
        return cls


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

    def add(self, t):
        """
        Store one experience into the buffer.

        Args:
            exp(Experience): the experience to store in the buffer.
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.last = (self.last + 1) % self.capacity
        self.buffer[self.last] = deepcopy(t)

    def sample(self, num_samples, is_episode_end_f, num_seqs=0):
        """
        Generate a batch of Samples. Each Sample represents a sequence of
        Experiences (length>=1). And a sequence must not cross the boundary
        between two games. 

        Args:
            num_samples(int): Number of samples to generate.
            
        Returns: A generator of Samples
        """
        exp_seqs = []
        if len(self.buffer) <= 1000:
            return exp_seqs

        if num_seqs == 0:
            num_seqs = num_samples
        for _ in xrange(num_seqs):
            while True:
                idx = np.random.randint(0, len(self.buffer) - 1)
                if not self.buffer_end(idx) and not is_episode_end_f(
                        self.buffer[idx]):
                    break
            indices = [idx]
            for i in range(num_samples / num_seqs):
                if self.buffer_end(idx) or is_episode_end_f(self.buffer[idx]):
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

    def sample(self, is_episode_end_f):
        exp_seqs = []
        while len(self.q) > 1:
            exps = []
            while len(self.q) > 1 and not is_episode_end_f(self.q[0]):
                # no need to deepcopy here as the selected exp is immediately
                # removed from the queue
                exps.append(self.q.popleft())
            if (exps):
                exps.append(deepcopy(self.q[0]))
                exp_seqs.append(exps)
            if is_episode_end_f(self.q[0]):
                self.q.popleft()
        return exp_seqs
