import unittest
from flare.env_zoo.xworld import XWorldEnv
import numpy as np
from random import randint


class TestXWorld(unittest.TestCase):
    def __init__(self, name="TestXWorld"):
        unittest.TestCase.__init__(self, name)
        options = {
            "x3_conf": "./walls3d.json",
            "context": 3,
            "pause_screen": True,
            "x3_training_img_width": 64,
            "x3_training_img_height": 64,
            "x3_big_screen": True,
            "color": True
        }

        with open("./dict.txt") as f:
            word_list = f.readlines()
            word_list = [word.strip() for word in word_list]
        self.env = XWorldEnv("xworld3d", options, word_list)

    def test_reset(self):
        ob = self.env.reset()
        # the channel num = color x context = 3 x 3
        self.assertEqual(ob["screen"].shape, (9, 64, 64))

        self.assertIsInstance(ob["sentence"], list)
        self.assertIsInstance(ob["sentence"][0], np.ndarray)

    def test_step(self):
        self.env.reset()
        dims = self.env.action_dims()
        a = np.array([randint(0, dims["action"] - 1)])
        sentence = [
            np.array([randint(0, dims["pred_sentence"] - 1)])
            for i in range(10)
        ]
        states, rewards, next_game_over = self.env.step(
            dict(
                action=a, pred_sentence=sentence))


if __name__ == "__main__":
    unittest.main()
