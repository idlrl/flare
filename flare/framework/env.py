import gym
import numpy as np
from scipy import misc
from abc import ABCMeta, abstractmethod
from py_simulator import Simulator
"""
A file for wrapping different environments to have a unified API interface
used by Agent's control flow.
"""

class Env(object):
    """
    An abstract class for environment. A new environment should inherit this
    class
    """
    
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def reset(self):
        """
        reset the environment and return the initial observation
        """
        pass
    
    @abstractmethod
    def step(self, actions, actrep):
        """
        Given a list of ordered actions, forward the environment actrep step.
        The output should be a list of next observations, a list of rewards,
        and the next game over.
        """
        pass
    
    @abstractmethod
    def observation_dims(self):
        """
        Return a list of lists as observation dimensions, each list for one 
        observation.
        Each list contains the dimension numbers of that input.
        """
        pass
    
    @abstractmethod
    def action_dims(self):
        """
        Return a list of integers as action dimensions, each integer for an action.
        For each integer,
        if the corresponding action is discrete, then it means
        the total number of actions;
        if continous, then it means the length of the action vector.
        if language, then it means the cardinality of the dictionary
        """
        pass
    
    @abstractmethod
    def time_out(self):
        """
        Return a boolean of whether the env has timed out 
        """
        pass


class XWorldEnv(Env):
    """
    A wrapper for XWorld env.
    """
    
    def __init__(self, game_name, contexts=1, options=None, dict=None):
        """
        options: see test_xworld3d.py in XWorld for an example of options
        dict: dictionary for the langauge part. key=id, value=word
        """
        self.env = Simulator.create(game_name, options)
        assert contexts >= 1, "contexts must be a positive number!"
        self.contexts = contexts
        self.buf = None
        self.dict_id2w = dict
        self.dict_w2id = {v: k for k, v in dict.iteritems()}
        self.height, self.width, self.channels = \
                                        self.env.get_screen_out_dimensions()
    
    
    def preprocess_observation(self, ob):
        screen = ob['screen'].reshape([self.channels, self.height, self.width])
        sentence = [np.array(self.dict_w2id(word)) for word in 
                        ob['sentence'].split(" ")]
        
        return screen, sentence
    
    def reset(self):
        ## should return a list of observations
        self.env.reset_game()
        init_ob = self.env.get_state()
        screen, sentence = self.preprocess_observation(init_ob) 

        self.buf = [np.zeros(screen.shape).astype(screen.dtype) \
                    for i in range(self.contexts - 1)]
        ## the newest state is the last element
        self.buf.append(screen)
        
        ## concat along the channel dimension
        return [np.concatenate(self.buf), sentence]


    def step(self, actions, actrep=1):
        assert len(actions) == 2, "xworld requires two actions, one action, \
                                   one language"
        a = actions[0]
        if "int" in str(a.dtype):
            a = a[0]
        
        sentence = " ".join([self.dict_id2w(id) for id in actions[1]])
        
        total_reward = self.env.take_actions({"action": a, 
                                               "pred_sentence": sentence}, 
                                               actrep,
                                               False)
        
        next_game_over = self.env.game_over() != "alive"
        next_ob = self.env.get_state()

        screen, sentence = self.preprocess_observation(next_ob)
        self.buf.append(screen)
        if len(self.buf) > self.contexts:
            self.buf.pop(0)
        return [np.concatenate(self.buf), sentence], \
               [total_reward], next_game_over


    def observation_dims(self):
        screen_shape = [self.contexts*self.channels, self.height, self.width]
        sentence_shape = [len(self.dict_id2w.keys())]
        
        return [screen_shape, sentence_shape]

    def action_dims(self):
        num_actions = self.env.get_num_actions()
        return [num_actions, len(self.dict_id2w.keys())]
    
    def time_out(self):
        return "max_step" in self.env.game_over()


class GymEnv(Env):
    """
    A convenient wrapper for OpenAI Gym env. Used to unify the
    environment interfaces.
    """

    def __init__(self, game_name, contexts=1, options=None):
        self.gym_env = gym.make(game_name)
        assert contexts >= 1, "contexts must be a positive number!"
        self.contexts = contexts
        self.buf = None
        self.steps = 0

    def reset(self):
        self.steps = 0
        ## should return a list of observations
        init_ob = self.preprocess_observation(self.gym_env.reset())
        self.buf = [np.zeros(init_ob.shape).astype(init_ob.dtype) \
                    for i in range(self.contexts - 1)]
        ## the newest state is the last element
        self.buf.append(init_ob)
        ## concat along the channel dimension
        return [np.concatenate(self.buf)]
    
    def time_out(self):
        return self.steps >= self.gym_env._max_episode_steps - 1
    
    def render(self):
        self.gym_env.render()

    def step(self, actions, actrep=1):
        """
        Given a list of ordered actions, forward the environment one step.
        The output should be a list of next observations, a list of rewards,
        and the next game over.
        """
        assert len(actions) == 1, "OpenAI Gym only accepts a single action!"
        a = actions[0]
        if "int" in str(a.dtype):
            a = a[0]  ## Gym accepts scalars for discrete actions

        total_reward = 0
        next_game_over = False
        for i in range(actrep):
            self.steps += 1
            next_ob, reward, game_over, _ = self.gym_env.step(a)
            total_reward += reward
            next_game_over = next_game_over | game_over

        next_ob = self.preprocess_observation(next_ob)
        self.buf.append(next_ob)
        if len(self.buf) > self.contexts:
            self.buf.pop(0)
        return [np.concatenate(self.buf)], [total_reward], next_game_over

    def preprocess_observation(self, ob):
        """
        A preprocssing function that transforms every observation before
        feeding it to the model. The preprocessing could be normalization,
        image resizing, etc.
        By default, there is no preprocessing. The user can inherit this base
        class and override this function. For example, see GymEnvImage.
        """
        return ob.astype("float32")

    def observation_dims(self):
        shape = self.gym_env.observation_space.shape
        ## only the first dim has several contexts
        ## Gym has a single input
        return [[shape[0] * self.contexts] + list(shape[1:])]

    def action_dims(self):
        """
        Gym has a single action.
        """
        act_space = self.gym_env.action_space
        if isinstance(act_space, gym.spaces.Discrete):
            return (act_space.n, )
        return act_space.shape


class GymEnvImage(GymEnv):
    """
    A derived class that overrides the self.preprocess_observation function
    so that RGB image input is resized, converted to gray scale, or rescaled
    to a proper value range.
    """

    def __init__(self, game_name, contexts=1, height=-1, width=-1, gray=False):
        super(GymEnvImage, self).__init__(game_name, contexts)
        self.height = height
        self.width = width
        self.gray = gray

    def preprocess_observation(self, ob):
        assert isinstance(ob, np.ndarray)
        assert ob.dtype == "uint8", "The observation must be uint8 (0-255)!"
        assert len(ob.shape) <= 3
        if self.height > 0:
            ob = misc.imresize(ob, (self.height, self.width))
        if self.gray and len(ob.shape) == 3 and ob.shape[-1] > 1:
            ## simply average across the three channels
            ## but still keep the number of axes the same
            ob = np.expand_dims(np.mean(ob, axis=-1), axis=-1)
        ## reshape HxWxC to CxHxW
        ob = np.moveaxis(ob, -1, 0)
        ## we should convert ob to a floating array in [0,1]
        return ob.astype("float32") / 255.0

    def observation_dims(self):
        h, w, d = self.gym_env.observation_space.shape
        if self.gray:
            d = 1
        d *= self.contexts
        if self.height > 0:
            return [(d, self.height, self.width)]
        return [(d, h, w)]
