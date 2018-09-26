from py_simulator import Simulator
from flare.framework.env import Env
import numpy as np


class XWorldEnv(Env):
    """
    A wrapper for XWorld env that will be created only once in a process.
    The OpenGL context needed by XWorld requires that only one instance of this
    class can be created in a process.
    """

    def __init__(self,
                 game_name,
                 options,
                 word_list,
                 opengl_init=True,
                 show_frame=False):
        """
        options: see test_xworld3d.py in XWorld for an example of options
        word_list: a list of words used in the language part
        """
        assert game_name == "xworld" or game_name == "xworld3d", \
            "Incorrect name provided!"
        options["x3_opengl_init"] = opengl_init
        self.env = Simulator.create(game_name, options)
        self.show_frame = show_frame
        self.dict_id2w = {id: word for id, word in enumerate(word_list)}
        self.dict_w2id = {v: k for k, v in self.dict_id2w.iteritems()}
        self.height, self.width, self.channels, self.contexts = \
                                        self.env.get_screen_out_dimensions()
        self.num_actions = self.env.get_num_actions()
        self.input_key1 = "screen"
        self.input_key2 = "sentence"
        self.input_key3 = "prev_action"
        self.action_key1 = "action"
        self.action_key2 = "pred_sentence"
        self.reward_key = "reward"

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
        ## should return a dictionary of observations
        self.env.reset_game()
        init_ob = self.env.get_state()
        screen, sentence = self.preprocess_observation(init_ob)

        return {
            self.input_key1: screen,
            self.input_key2: sentence,
            self.input_key3: [0]
        }

    def step(self, actions, actrep=1):
        """
        actions: a dictionary of length 2
        actions["action"]: np.array, shape = [1], represent action
        actions["pred_sentence"] (optional): list of np.array, each array shape = [1],
                                             represent langauge
        """
        ## copy because we will change actions values; shallow copy is good enough
        act = actions.copy()
        assert len(
            act) <= 2, "xworld requires at most two actions, one action, \
                                   one language"

        a = act[self.action_key1]
        assert "int" in str(a.dtype)
        ## xworld3d accepts discrete actions as int32
        act[self.action_key1] = a[0].astype("int")

        if self.action_key2 in act:
            act[self.action_key2] = " ".join([self.dict_id2w[id[0]] \
                                                 for id in act[self.action_key2]])
        else:
            act[self.action_key2] = ""

        total_reward = self.env.take_actions(act, actrep, self.show_frame)

        if self.env.game_over() == "success":
            next_game_over = 1
        elif self.env.game_over() == "dead" or self.env.game_over(
        ) == "lost_life":
            next_game_over = -1
        else:  ## "max_step" timeout
            next_game_over = 0

        next_ob = self.env.get_state()

        screen, sentence = self.preprocess_observation(next_ob)
        return {self.input_key1 : screen,
                self.input_key2 : sentence,
                self.input_key3 : [act[self.action_key1]]}, \
            {self.reward_key : [total_reward]}, next_game_over

    def observation_dims(self):
        screen_shape = (self.contexts * self.channels, self.height, self.width)
        sentence_shape = (len(self.dict_id2w.keys()), )
        return {
            self.input_key1: screen_shape,
            self.input_key2: sentence_shape
        }  ## self.input_key3 unnecessary

    def action_dims(self):
        return {
            self.action_key1: self.num_actions,
            self.action_key2: len(self.dict_id2w.keys())
        }

    def time_out(self):
        return "max_step" in self.env.game_over()
