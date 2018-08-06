import numpy as np
import unittest
from flare.common.error_handling import LastExpError
from flare.common.replay_buffer import Experience
from flare.common.replay_buffer import ReplayBuffer, NoReplacementQueue, Sample


class TestNoReplacementQueue(unittest.TestCase):
    @classmethod
    def is_episode_end(cls, t):
        return t[3][0]

    def test_sampling(self):
        exp_q = NoReplacementQueue()
        #          obs           r    a    e
        exp_q.add((np.zeros(10), [1], [1], [0]))
        exp_q.add((np.zeros(10), [0], [-1], [1]))  # 1st episode end
        exp_q.add((np.zeros(10), [1], [2], [0]))
        exp_q.add((np.zeros(10), [1], [3], [0]))
        exp_q.add((np.zeros(10), [1], [4], [0]))
        exp_seqs = exp_q.sample(self.is_episode_end)
        self.assertEqual(len(exp_q), 1)
        self.assertEqual(len(exp_seqs), 2)
        self.assertEqual(len(exp_seqs[0]), 2)
        self.assertEqual(exp_seqs[0][0][2], [1])
        self.assertEqual(exp_seqs[0][1][2], [-1])
        self.assertEqual(len(exp_seqs[1]), 3)
        self.assertEqual(exp_seqs[1][0][2], [2])
        self.assertEqual(exp_seqs[1][1][2], [3])
        self.assertEqual(exp_seqs[1][2][2], [4])
        #          obs           r    a    e
        exp_q.add((np.zeros(10), [0], [-2], [1]))
        exp_seqs = exp_q.sample(self.is_episode_end)
        self.assertEqual(len(exp_q), 0)
        self.assertEqual(len(exp_seqs), 1)
        self.assertEqual(len(exp_seqs[0]), 2)
        self.assertEqual(exp_seqs[0][0][2], [4])
        self.assertEqual(exp_seqs[0][1][2], [-2])
        self.assertEqual(len(exp_q), 0)


class TestReplayBuffer(unittest.TestCase):
    @classmethod
    def is_episode_end(cls, t):
        return t[3]

    def test_single_instance_replay_buffer(self):
        capacity = 30
        episode_len = 4
        buf = ReplayBuffer(capacity)
        for i in xrange(10 * capacity):
            #        obs           r      a  e
            buf.add((np.zeros(10), i * 0.5, i, (i + 1) % episode_len == 0))
            # check the circular queue in the buffer
            self.assertTrue(len(buf) == min(i + 1, capacity))
            if (len(buf) < 2):  # need at least two elements
                continue
            # should raise error when trying to pick up the last element
            exp_seqs = buf.sample(capacity, self.is_episode_end, 0)
            for exp_seq in exp_seqs:
                self.assertEqual(len(exp_seq), 2)
                self.assertNotEqual(exp_seq[0][3], 1)
                self.assertEqual(exp_seq[1][2], exp_seq[0][2] + 1)


if __name__ == '__main__':
    unittest.main()
