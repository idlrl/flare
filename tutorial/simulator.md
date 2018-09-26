# Link a Simulator to FLARE
To link a simulator/environment to FLARE, the user needs to derive from the base class `Env` in `<flare_root>/flare/framework/env.py` and implement a standard set of five functions:

```python
@abstractmethod
def reset(self):
    """
    Reset the environment and return a dictionary of initial observations
    """
    pass

@abstractmethod
def step(self, actions, actrep):
    """
    Given a dictionary of actions, forward the environment actrep step.
    The output should be a dictionary of next observations, a dictionary of
    reward vectors (each vector for a kind), and next_game_over which only has
    three possible values: 0 alive, 1 success, -1 failure/dead
    """
    pass

@abstractmethod
def observation_dims(self):
    """
    Return a dictionary of tuples as observation dimensions, each tuple for one
    observation.
    Each tuple contains the dimension numbers of that input.
    """
    pass

@abstractmethod
def action_dims(self):
    """
    Return a dictionary of integers as action dimensions, each integer for an
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
```

The comments above are already clear about what to be expected for the outputs of each of the five functions. A few example environments can be found in `<flare_root>/flare/env_zoo`. Some explanations:

* We expect the simulator to return a dictionary of observations in `step`. This is a generalization to most existing simulators where only one observation is available for the agent every step. The reason for this generalization is that an embodied agent could have multiple channels of perception, e.g., vision, language, depth sensing, etc. As a result, we also assume that a dictionary of observations are fed to the agent model. Each observation could have different shapes: the observation could be a flattend vector or an N-D matrix. Thus `observation_dims` should return a dictionary of tuples where a tuple contains several dimension numbers.
* Similarly, the agent always produces a dictionary of actions instead of a single action every step. The actions can be in different forms, e.g., sentences, movements, etc. We expect the simulator to take a dictionary of actions and step forward.
* After each step, the simulator could return a dictionary of reward vectors, because generally the environment gives multiple reward signals to the agent. The algorithm/model will decide how to exploit the various rewards (e.g., linear combination, weighting, independent training, etc).
* It is critical to ensure that the gameover status returned by `step` only indicates death/failure of the agent, but not timeout. Timeout will be handled by the `time_out` function. Death/failure and timeout should be strictly differentiated from each other.
* It is suggested that the simulator *preprocesses* the input data and *postprocesses* the output data for the agent. The agent will take the inputs as is and produce outputs from its model as is. For example, if the input is an uint8 image, then the simulator should divide it by 255 in order for the data to fall in a reasonable range. Another example is that if the simulator returns a sentence as an input, then it should first convert it to a sequence of IDs by looking up the dictionary. On the other hand, if the agent speaks by outputting a sequence of IDs, then the simulator should convert it to a string by reversely looking up the dictionary.

***NOTE:*** *The keys defined in various [Model](modules.md) specs should match the keys of the input/output dictionaries of the above five functions.*

We have linked three environments (`GymEnv`, `GymEnvImage`, and `XWorldEnv`) to FLARE for the user in `<flare_root>/flare/env_zoo`.
