from abc import ABCMeta, abstractmethod
"""
A file for wrapping different environments to have a unified API interface
used by Agent's control flow.
"""


class Env(object):
    """
    An abstract class for environment. A new environment should inherit from
    this class
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def reset(self):
        """
        reset the environment and return a list of initial observations
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
        Return a list of tuples as observation dimensions, each list for one 
        observation.
        Each list contains the dimension numbers of that input.
        """
        pass

    @abstractmethod
    def action_dims(self):
        """
        Return a tuple of integers as action dimensions, each integer for an 
        action. For each integer, if the corresponding action is discrete, 
        then it means the total number of actions;
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
