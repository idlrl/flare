import gym
"""
A file for wrapping different environments to have a unified API interface
used by Agent's control flow.
"""


class GymEnv(object):
    """
    A convenient wrapper for OpenAI Gym env. Used to unify the
    environment interfaces.
    """

    def __init__(self, game_name):
        self.gym_env = gym.make(game_name)

    def reset(self):
        ## should return a list of observations
        return [self.gym_env.reset()]

    def get_max_steps(self):
        return self.gym_env._max_episode_steps

    def render(self):
        self.gym_env.render()

    def step(self, actions):
        """
        Given a list of ordered actions, forward the environment one step.
        The output should be a list of next observations, a list of rewards,
        and the next game over.
        """
        assert len(actions) == 1, "OpenAI Gym only accepts a single action!"
        a = actions[0]
        if "int" in str(a.dtype):
            a = a[0]  ## Gym accepts scalars for discrete actions
        next_ob, reward, next_game_over, _ = self.gym_env.step(a)
        return [next_ob], [reward], next_game_over

    @property
    def observation_space(self):
        return self.gym_env.observation_space

    @property
    def action_space(self):
        return self.gym_env.action_space
