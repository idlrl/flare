import torch
import torch.nn as nn
from abc import ABCMeta, abstractmethod


def check_duplicate_spec_names(model):
    """
    Check if there are two specs that have the same name.
    """
    specs = model.get_input_specs() \
            + model.get_action_specs() \
            + model.get_state_specs() \
            + model.get_reward_specs()
    names = [name for name, _ in specs]
    duplicates = set([n for n in names if names.count(n) > 1])
    assert not duplicates, \
        "duplicate names with different specs: " + " ".join(duplicates)


class Model(nn.Module):
    """
    A Model is owned by an Algorithm. It implements the entire network model of
    a specific problem.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Model, self).__init__()

    @abstractmethod
    def get_input_specs(self):
        """
        Output: list of '(name props)' tuples
                where props is a dict that must contain the 'shape' keyword
        """
        pass

    def get_state_specs(self):
        """
        States are optional to a Model.
        Output: list of '(name props)' tuples
                where props is a dict that must contain the 'shape' keyword
        """
        return []

    @abstractmethod
    def get_action_specs(self):
        """
        Output: list of '(name props)' tuples
                where props is a dict that must contain the 'shape' keyword
        """
        pass

    def get_reward_specs(self):
        """
        By default, a scalar reward.
        User can specify a vector of rewards for some problems

        Output: list of '(name props)' tuples
                where props is a dict that must contain the 'shape' keyword
        """
        return [("reward", dict(shape=[1]))]

    def policy(self, inputs, states):
        """
        Return: action_dicts: a dict of action distribution objects
                states
                An action distribution object can be created with
                PolicyDistribution().
        Optional: a model might not always have to implement policy()
        """
        raise NotImplementedError()

    def value(self, inputs, states):
        """
        Return: values: a dict of estimated values for the current observations and states
                        For example, "q_value" and "v_value"
        Optional: a model might not always have to implement value()
        """
        raise NotImplementedError()


class Algorithm(object):
    """
    An Algorithm implements two functions:
    1. predict() computes forward
    2. learn() computes a cost for optimization

    An algorithm should be only part of a network. The user only needs to
    implement the rest of the network in the Model class.
    """

    def __init__(self, model, gpu_id, iterations_per_batch=1):
        """
        iterations_per_batch: how many iterations of forwardbackward
        are performed on every sampled batch.
        Only set it greater than 1 if you are aware of off-policy training.
        """
        assert isinstance(model, Model)
        check_duplicate_spec_names(model)
        self.model = model
        if torch.cuda.is_available() and gpu_id >= 0:
            self.device = "cuda:" + str(gpu_id)
        else:
            self.device = "cpu"
        self.model.to(self.device)
        assert iterations_per_batch > 0
        self.iterations_per_batch = iterations_per_batch

    def get_input_specs(self):
        return self.model.get_input_specs()

    def get_state_specs(self):
        return self.model.get_state_specs()

    def get_action_specs(self):
        """
        For non-RL algorithms, this can return []
        """
        return self.model.get_action_specs()

    def get_reward_specs(self):
        """
        For non-RL algorithms, this can return []
        """
        return self.model.get_reward_specs()

    def predict(self, inputs, states):
        """
        Given the inputs and states, this function does forward prediction and updates states.
        Input: inputs(dict), states(dict)
        Output: actions(dict), states(dict)

        Optional: an algorithm might not implement predict()
        """
        pass

    def _rl_predict(self, behavior_model, inputs, states):
        """
        Given a behavior model (not necessarily equal to self.model), this function
        performs a normal RL prediction according to inputs and states.
        A behavior model different from self.model indicates off-policy training.

        The user can choose to call this function for convenience.
        """
        distributions, states = behavior_model.policy(inputs, states)
        actions = {}
        for key, dist in distributions.iteritems():
            actions[key] = dist.sample()
            prob_key = key + "_log_prob"
            actions[prob_key] = dist.log_prob(actions[key]).unsqueeze(-1)
            if len(actions[key].size()) == 1:  ## for discrete actions
                actions[key] = actions[key].unsqueeze(-1)
            # else for continuous actions, each action is already a vector
        return actions, states

    def learn(self, inputs, next_inputs, states, next_states, next_alive,
              actions, next_actions, rewards):
        """
        This function computes a learning cost to be optimized.
        The return should be the cost and updated states.
        Output: cost(dict), states(dict)

        Optional: an algorithm might not implement learn()
        """
        pass
