import os
import glob
import glog
import torch
import torch.optim as optim
from algorithm import Model, Algorithm
import recurrent as rc
import numpy as np
import operator
from flare.framework.computation_data_processor import ComputationDataProcessor
from multiprocessing import Value


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
                 model_dir="",
                 pass_num=0,
                 **kwargs):
        assert isinstance(algorithm, Algorithm)
        self.name = name
        if model_dir == "":
            self.model_dir = ""
        else:
            os.system("mkdir -p " + model_dir)
            self.model_dir = model_dir
        self.alg = algorithm
        self._cdp_args = kwargs
        self._cdp = None
        ## if model_dir is not empty, then we load an init model
        if self.model_dir != "":
            if glob.glob(self.model_dir + "/*"):
                if pass_num > 0:
                    model_file = self.model_dir + ("/%06d/%s.w" %
                                                   (pass_num, self.name))
                else:
                    ## otherwise we take the most recent model
                    model_file = self.model_dir + "/lastest/" + self.name + ".w"
                self.alg.model.load_state_dict(torch.load(model_file))
                glog.info("CT[%s] model loaded from '%s'" %
                          (self.name, model_file))
        self.model_save_signal = Value('i', -1)

    def save_model(self, idx):
        if self.model_dir != "":
            model_file = "%s/%06d/%s.w" % (self.model_dir, idx, self.name)
            lastest_model_file = "%s/lastest/%s.w" % (self.model_dir,
                                                      self.name)
            ## create a directory for the current pass
            os.system("mkdir -p " + os.path.dirname(model_file))
            os.system("mkdir -p " + os.path.dirname(lastest_model_file))
            torch.save(self.alg.model.state_dict(), model_file)
            torch.save(self.alg.model.state_dict(), lastest_model_file)
            glog.info("CT[%s] model saved to '%s'" % (self.name, model_file))

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
            self._cdp = ComputationDataProcessor(
                name=self.name, ct=self, **self._cdp_args)
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
            pred_actions, next_states = self.alg.predict(inputs, states)
        pred_actions = self._retrieve_np_arrays(pred_actions)
        next_states = self._retrieve_np_arrays(next_states)

        ## these are the action and state names expected in the outputs of predict()
        action_names = sorted(
            [name for name, _ in self.alg.get_action_specs()])
        state_names = sorted([name for name, _ in self.alg.get_state_specs()])
        return pred_actions, next_states

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

        if self.model_save_signal.value >= 0:
            self.save_model(self.model_save_signal.value)
            self.model_save_signal.value = -1

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

        for i in range(self.alg.iterations_per_batch):
            costs, _, _ = self.alg.learn(inputs, next_inputs, states,
                                         next_states, next_alive, actions,
                                         next_actions, rewards)
        return self._retrieve_np_arrays(costs)
