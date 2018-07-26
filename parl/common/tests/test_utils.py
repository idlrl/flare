import numpy as np
import unittest
from parl.common.error_handling import LastExpError
from parl.common.utils import concat_dicts, split_dict


class TestUtils(unittest.TestCase):
    def test_concat_and_split_numpy(self):
        d1 = dict(sensors=np.random.rand(3, 10), states=np.random.rand(3, 20))
        d2 = dict(sensors=np.random.rand(2, 10), states=np.random.rand(2, 20))
        D = concat_dicts([d1, d2])
        starts = [0, 3, 5]
        self.assertTrue(np.array_equal(D["sensors"][0:3], d1["sensors"]))
        self.assertTrue(np.array_equal(D["sensors"][3:], d2["sensors"]))
        self.assertTrue(np.array_equal(D["states"][0:3], d1["states"]))
        self.assertTrue(np.array_equal(D["states"][3:], d2["states"]))

        dd1, dd2 = split_dict(D, starts)
        self.assertEqual(dd1.viewkeys(), dd2.viewkeys())
        for k in dd1.iterkeys():
            self.assertTrue(np.array_equal(dd1[k], d1[k]))
            self.assertTrue(np.array_equal(dd2[k], d2[k]))

    def test_concat_and_split_nested_list(self):
        d1 = dict(
            sensors=[
                [[0, 1], [2, 3], [4, 5]],  # sequence 1
                [[6, 7], [8, 9]],  # sequence 2
                [[10, 11]]
            ],  # sequence 3
            states=[[0.0, 0.1, 0.2], [1.0, 1.1, 1.2], [2.0, 2.1, 2.2]])
        d2 = dict(
            sensors=[[[12, 13], [14, 15], [16, 17], [18, 19]], [[20, 21]]],
            states=[[3.0, 3.1, 3.2], [4.0, 4.1, 4.2]])
        D = concat_dicts([d1, d2])
        starts = [0, 3, 5]
        self.assertTrue(np.array_equal(D["sensors"][0:3], d1["sensors"]))
        self.assertTrue(np.array_equal(D["sensors"][3:], d2["sensors"]))
        self.assertTrue(np.array_equal(D["states"][0:3], d1["states"]))
        self.assertTrue(np.array_equal(D["states"][3:], d2["states"]))

        dd1, dd2 = split_dict(D, starts)
        self.assertEqual(dd1.viewkeys(), dd2.viewkeys())
        for k in dd1.iterkeys():
            self.assertTrue(np.array_equal(dd1[k], d1[k]))
            self.assertTrue(np.array_equal(dd2[k], d2[k]))


if __name__ == '__main__':
    unittest.main()
