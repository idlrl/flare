import torch
import numpy as np
from flare.framework.algorithm import Model, Algorithm


def split_list(l, sizes):
    """
    Split a list into several chunks, each chunk with a size in sizes
    """
    chunks = []
    offset = 0
    for size in sizes:
        chunks.append(l[offset:offset + size])
        offset += size
    return chunks


class ComputationTask(object):
    """
    A ComputationTask is responsible for the general data flow
    outside the algorithm

    A ComputationTask is created in a bottom-up way:
    a. create a Model
    b. create an Algorithm with the model as an input
    c. define a ComputationTask with the algorithm
    """

    def __init__(self, algorithm):
        assert isinstance(algorithm, Algorithm)
        self.alg = algorithm
        if torch.cuda.is_available() and self.alg.gpu_id >= 0:
            self.device = torch.device("cuda:" + str(self.alg.gpu_id))
        else:
            self.device = torch.device("cpu")
        ## put the model on the device
        self.alg.model.to(self.device)

    def _create_tensors(self, np_arrays_dict, specs):
        ## We want to convert numpy arrayes to torch tensors,
        ## and put them on the device
        tensors = {}
        for name, props in specs:
            assert name in np_arrays_dict, "keyword %s does not exist in np arrays!" % name

            np_array = np_arrays_dict[name]
            assert isinstance(np_array, np.ndarray)
            dtype = ("float32" if "dtype" not in props else props["dtype"])
            assert np_array.dtype == dtype, "%s %s" % (np_array.dtype, dtype)
            assert "shape" in props, "You must specify the tensor shape in the specs!"
            assert tuple(np_array.shape[1:]) == tuple(props["shape"]), \
                "%s %s" % (tuple(np_array.shape[1:]), tuple(props["shape"]))

            tensors[name] = torch.from_numpy(np_array)
            tensors[name].requires_grad_()
            tensors[name].to(self.device)

        return tensors

    def _retrieve_np_arrays(self, tensors_dict):
        return {
            name: t.detach().numpy()
            for name, t in tensors_dict.iteritems()
        }

    def predict(self, inputs, states=dict()):
        """
        ComputationTask predict API
        This function is responsible to convert Python data to Fluid tensors, and
        then convert the computational results in the reverse way.
        """
        inputs = self._create_tensors(inputs, self.alg.get_input_specs())
        states = self._create_tensors(states, self.alg.get_state_specs())
        with torch.no_grad():
            pred_actions, pred_states = self.alg.predict(inputs, states)
        pred_actions = self._retrieve_np_arrays(pred_actions)
        pred_states = self._retrieve_np_arrays(pred_states)

        ## these are the action and state names expected in the outputs of predict()
        action_names = sorted(
            [name for name, _ in self.alg.get_action_specs()])
        state_names = sorted([name for name, _ in self.alg.get_state_specs()])
        assert sorted(pred_actions.keys()) == action_names
        assert sorted(pred_states.keys()) == state_names
        return pred_actions, pred_states

    def learn(self,
              inputs,
              next_inputs,
              next_episode_end,
              actions,
              rewards,
              states=dict(),
              next_states=dict()):
        """
        ComputationTask learn API
        This function is responsible to convert Python data to Fluid tensors, and
        then convert the computational results in the reverse way.
        """

        def _get_next_specs(specs):
            return [("next_" + spec[0], spec[1]) for spec in specs]

        inputs = self._create_tensors(inputs, self.alg.get_input_specs())
        next_inputs = self._create_tensors(
            next_inputs, _get_next_specs(self.alg.get_input_specs()))
        states = self._create_tensors(states, self.alg.get_state_specs())
        next_states = self._create_tensors(
            next_states, _get_next_specs(self.alg.get_state_specs()))
        next_episode_end = self._create_tensors(
            next_episode_end, [("next_episode_end", dict(shape=[1]))])
        actions = self._create_tensors(actions, self.alg.get_action_specs())
        rewards = self._create_tensors(rewards, self.alg.get_reward_specs())

        costs = self.alg.learn(inputs, next_inputs, states, next_states,
                               next_episode_end, actions, rewards)
        costs = self._retrieve_np_arrays(costs)
        return costs
