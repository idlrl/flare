import os
import glog
import glob
import torch
import torch.optim as optim
from algorithm import Model, Algorithm
import recurrent as rc
import numpy as np
import operator
from flare.framework.computation_data_processor import ComputationDataProcessor


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

    def __init__(self,
                 name,
                 algorithm,
                 hyperparas=dict(
                     lr=1e-4, grad_clip=0),
                 model_dir="",
                 **kwargs):
        assert isinstance(algorithm, Algorithm)
        self.name = name
        self.hp = hyperparas
        if model_dir == "":
            self.model_file = ""
        else:
            os.system("mkdir -p " + model_dir)
            self.model_file = model_dir + "/" + name
        self.alg = algorithm
        self.optim = optim.RMSprop(
            self.alg.model.parameters(), lr=hyperparas["lr"])
        self._cdp_args = kwargs
        self._cdp = None
        ## if model_file is not empty, then we load an init model
        if self.model_file != "":
            models = glob.glob(self.model_file)
            if models:
                ## for now, we take the most recent model
                most_recent_model_file = sorted(models)[-1]
                self.alg.model.load_state_dict(
                    torch.load(most_recent_model_file))
                glog.info("Loading the model " + most_recent_model_file)

    def save_model(self, idx):
        if self.model_file != "":
            model_file = "%s%06d" % (self.model_file, idx)
            ## TODO: CUDA error
            torch.save(self.alg.model.state_dict(), model_file)

    def get_state_specs(self):
        return self.alg.get_state_specs()

    def get_input_specs(self):
        return self.alg.get_input_specs()

    def get_action_specs(self):
        return self.alg.get_action_specs()

    def get_reward_specs(self):
        return self.alg.get_reward_specs()

    @property
    def CDP(self):
        if self._cdp is None:
            self._cdp = ComputationDataProcessor(self.name, self,
                                                 **self._cdp_args)
        return self._cdp

    def _create_tensors(self, arrays_dict, specs):
        ## We want to convert python arrays to a hierarchy of torch tensors,
        ## and put them on the device
        tensors = {}
        if arrays_dict is None:
            return {}
        for name, props in specs:
            assert name in arrays_dict, "keyword %s does not exist in python arrays!" % name
            array = arrays_dict[name]
            dtype = ("float32" if "dtype" not in props else props["dtype"])
            assert "shape" in props, "You must specify the tensor shape in the specs!"
            tensors[name] = rc.make_hierarchy_of_tensors(
                array, dtype, self.alg.device, props["shape"])
        return tensors

    def _retrieve_np_arrays(self, tensors_dict):
        def numpy_recursion(ts):
            """
            Convert a hierarchy of tensors recursively to a hierarchy of
            numpy arrays.
            """
            if isinstance(ts, torch.Tensor):
                return ts.cpu().detach().numpy()
            else:
                assert isinstance(ts, list)
                return [numpy_recursion(t) for t in ts]

        return {
            name: numpy_recursion(t)
            for name, t in tensors_dict.iteritems()
        }

    def predict(self, inputs, states=None):
        """
        ComputationTask predict API
        This function is responsible to convert Python numpy arrays to pytorch
        tensors, and then convert the computational results in the reverse way.
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
        return pred_actions, pred_states

    def learn(self,
              inputs,
              next_inputs,
              next_alive,
              rewards,
              actions,
              next_actions=None,
              states=None,
              next_states=None):
        """
        ComputationTask learn API
        This function is responsible to convert Python numpy arrays to pytorch
        tensors, and then convert the computational results in the reverse way.
        """

        inputs = self._create_tensors(inputs, self.alg.get_input_specs())
        next_inputs = self._create_tensors(next_inputs,
                                           self.alg.get_input_specs())
        states = self._create_tensors(states, self.alg.get_state_specs())
        next_states = self._create_tensors(next_states,
                                           self.alg.get_state_specs())
        next_alive = self._create_tensors(
            next_alive, [("alive", dict(shape=[1]))])
        actions = self._create_tensors(actions, self.alg.get_action_specs())
        next_actions = self._create_tensors(next_actions,
                                            self.alg.get_action_specs())
        rewards = self._create_tensors(rewards, self.alg.get_reward_specs())

        def sum_cost(costs):
            if isinstance(costs, torch.Tensor):
                return (costs.view(-1).sum(), reduce(operator.mul,
                                                     costs.size()))
            assert isinstance(costs, list)
            costs, ns = zip(*map(sum_cost, costs))
            return sum(costs), sum(ns)

        for i in range(self.alg.iterations_per_batch):
            ## First we zero the gradients,
            ## after which backward() should be called by the user in algorithms' learn()
            self.optim.zero_grad()
            if states:  ## if states is not empty, we apply a recurrent_group first

                def outermost_step(*args):
                    ipts, nipts, nee, act, nact, rs, sts, nsts = split_list(
                        list(args), [
                            len(inputs), len(next_inputs), len(next_alive),
                            len(actions), len(next_actions), len(rewards),
                            len(states), len(next_states)
                        ])
                    ## We wrap each input into a dictionary because self.alg.learn
                    ## is expected to receive dicts and output dicts
                    costs, sts_update, nsts_update = self.alg.learn(
                        dict(zip(inputs.keys(), ipts)),
                        dict(zip(next_inputs.keys(), nipts)),
                        dict(zip(states.keys(), sts)),
                        dict(zip(next_states.keys(), nsts)),
                        dict(zip(next_alive.keys(), nee)),
                        dict(zip(actions.keys(), act)),
                        dict(zip(next_actions.keys(), nact)),
                        dict(zip(rewards.keys(), rs)))
                    self.cost_keys = costs.keys()
                    return costs.values(), \
                        [sts_update[k] for k in states.keys()] + \
                        [nsts_update[k] for k in next_states.keys()]

                costs = rc.recurrent_group(seq_inputs=inputs.values() + \
                                                 next_inputs.values() + \
                                                 next_alive.values() + \
                                                 actions.values() + \
                                                 next_actions.values() + \
                                                 rewards.values(),
                                           insts=[],
                                           init_states=states.values() + next_states.values(),
                                           step_func=outermost_step)
                costs = dict(zip(self.cost_keys, costs))
            else:
                costs, _, _ = self.alg.learn(inputs, next_inputs, states,
                                             next_states, next_alive, actions,
                                             next_actions, rewards)

            ## If gradient clipping is enabled, we should do this before step() after backward()
            if "grad_clip" in self.hp and self.hp["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(self.alg.model.parameters(),
                                               self.hp["grad_clip"])
            self.optim.step()

        return self._retrieve_np_arrays(costs)
