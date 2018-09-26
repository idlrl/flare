import gym
from flare.framework.env import Env
import numpy as np
from scipy import misc


class GymEnv(Env):
    """
    A convenient wrapper for OpenAI Gym env. Used to unify the
    environment interfaces.
    """

    def __init__(self, game_name, show_frame=False, contexts=1):
        self.gym_env = gym.make(game_name)
        assert contexts >= 1, "contexts must be a positive number!"
        self.contexts = contexts
        self.buf = None
        self.steps = 0
        self.show_frame = show_frame
        self.input_key = "sensor"
        self.action_key = "action"
        self.reward_key = "reward"

    def reset(self):
        self.steps = 0
        ## should return a dictionary of observations
        init_ob = self.preprocess_observation(self.gym_env.reset())
        self.buf = [np.zeros(init_ob.shape).astype(init_ob.dtype) \
                    for i in range(self.contexts - 1)]
        ## the newest state is the last element
        self.buf.append(init_ob)
        ## concat along the channel dimension
        return {self.input_key: np.concatenate(self.buf)}

    def time_out(self):
        return self.steps >= self.gym_env._max_episode_steps - 1

    def render(self):
        self.gym_env.render()

    def step(self, actions, actrep=1):
        """
        Given a dictionary of actions, forward the environment one step.
        The output should be a dictionary of next observations, a dictionary of
        reward vectors (each vector for a kind), and the next game over.
        """
        a = actions["action"]
        if "int" in str(a.dtype):
            a = a[0]  ## Gym accepts scalars for discrete actions

        total_reward = 0
        ## The Gym games do not differentiate between 'success' and 'failure'
        next_game_over = False
        self.steps += 1
        for i in range(actrep):
            next_ob, reward, game_over, _ = self.gym_env.step(a)
            total_reward += reward
            next_game_over = next_game_over | game_over
        if next_game_over:
            next_game_over = -1  ## when True, assume a failure

        next_ob = self.preprocess_observation(next_ob)
        self.buf.append(next_ob)
        if len(self.buf) > self.contexts:
            self.buf.pop(0)

        if self.show_frame:
            self.gym_env.render()

        return {self.input_key : np.concatenate(self.buf)}, \
            {self.reward_key : [total_reward]}, int(next_game_over)

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
        return {self.input_key: (shape[0] * self.contexts, ) + shape[1:]}

    def action_dims(self):
        """
        Gym has a single action.
        """
        act_space = self.gym_env.action_space
        if isinstance(act_space, gym.spaces.Discrete):
            return {self.action_key: act_space.n}
        return {self.action_key: act_space.shape[0]}


class GymEnvImage(GymEnv):
    """
    A derived class that overrides the self.preprocess_observation function
    so that RGB image input is resized, converted to gray scale, or rescaled
    to a proper value range.
    """

    def __init__(self,
                 game_name,
                 show_frame=False,
                 contexts=1,
                 height=-1,
                 width=-1,
                 gray=False):
        super(GymEnvImage, self).__init__(game_name, show_frame, contexts)
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
            return {self.input_key: (d, self.height, self.width)}
        return {self.input_key: (d, h, w)}
