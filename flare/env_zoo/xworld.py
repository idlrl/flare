from py_simulator import Simulator
from flare.framework.env import Env
import numpy as np
import pdb


class XWorldEnv(Env):
    """
    A wrapper for XWorld env.
    """

    def __init__(self, game_name, options=None, word_list=None):
        """
        options: see test_xworld3d.py in XWorld for an example of options
        word_list: a list of words used in the language part
        """
        self.env = Simulator.create(game_name, options)
        self.dict_id2w = {id: word for id, word in enumerate(word_list)}
        self.dict_w2id = {v: k for k, v in self.dict_id2w.iteritems()}
        self.height, self.width, self.channels = \
                                        self.env.get_screen_out_dimensions()

    def preprocess_observation(self, ob):
        # The XworldEnv handles context within itself. The screen output
        # already did the concatenation of recent context screens
        screen = np.array(ob['screen']).reshape([-1, self.height, self.width])
        sentence = [
            np.array([self.dict_w2id[word.lower()]])
            for word in ob['sentence'].split(" ")
        ]

        return screen, sentence

    def reset(self):
        ## should return a list of observations
        self.env.reset_game()
        init_ob = self.env.get_state()
        screen, sentence = self.preprocess_observation(init_ob)

        return [screen, sentence]

    def step(self, actions, actrep=1):
        """
        actions: a list of length 2
        actions[0]: np.array, shape = [1], represent action
        actions[1]: list of np.array, each array shape = [1], represent 
                    langauge
        """
        assert len(actions) == 2, "xworld requires two actions, one action, \
                                   one language"

        a = actions[0]
        if "int" in str(a.dtype):
            a = a[0]

        sentence = " ".join([self.dict_id2w[id[0]] for id in actions[1]])

        total_reward = self.env.take_actions({
            "action": a,
            "pred_sentence": sentence
        }, actrep, False)

        next_game_over = self.env.game_over() != "alive"
        next_ob = self.env.get_state()

        screen, sentence = self.preprocess_observation(next_ob)
        return [screen, sentence], \
               [total_reward], next_game_over

    def observation_dims(self):
        screen_shape = (self.contexts * self.channels, self.height, self.width)
        sentence_shape = (len(self.dict_id2w.keys()), )

        return [screen_shape, sentence_shape]

    def action_dims(self):
        num_actions = self.env.get_num_actions()
        return [num_actions, len(self.dict_id2w.keys())]

    def time_out(self):
        return "max_step" in self.env.game_over()
