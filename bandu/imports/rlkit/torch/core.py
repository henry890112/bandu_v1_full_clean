import abc
import numpy as np
from collections import OrderedDict

from torch import nn as nn
from torch.autograd import Variable

from bandu.imports.rlkit.torch import pytorch_util as ptu
from bandu.imports.rlkit.core.serializable import Serializable
from functools import reduce
import torch


class PyTorchModule(nn.Module, Serializable, metaclass=abc.ABCMeta):

    def get_param_values(self):
        return self.state_dict()

    def set_param_values(self, param_values):
        # new_param_values = param_values.copy()
        # try:
        self.load_state_dict(param_values)
        # except RuntimeError as e:
        #     import re
        #     e_str = re.search("(?<=state_dict: ).*(?=\.)", str(e)).group(0)
        #     e_str.replace(" ", "")
        #     x_stripped = [x.strip() for x in e_str.split(",")]
        #     for x in x_stripped:
        #         new_param_values[x] = nn.Parameter(torch.tensor(1.0))

    def get_param_values_np(self):
        state_dict = self.state_dict()
        np_dict = OrderedDict()
        for key, tensor in state_dict.items():
            np_dict[key] = ptu.get_numpy(tensor)
        return np_dict

    def set_param_values_np(self, param_values):
        torch_dict = OrderedDict()
        for key, tensor in param_values.items():
            torch_dict[key] = ptu.from_numpy(tensor)
        self.load_state_dict(torch_dict)

    def copy(self):
        copy = Serializable.clone(self)
        ptu.copy_model_params_from_to(self, copy)
        return copy

    def save_init_params(self, locals):
        """
        Should call this FIRST THING in the __init__ method if you ever want
        to serialize or clone this network.

        Usage:
        ```
        def __init__(self, ...):
            self.init_serialization(locals())
            ...
        ```
        :param locals:
        :return:
        """
        Serializable.quick_init(self, locals)

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["params"] = self.get_param_values()
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self.set_param_values(d["params"])

    def regularizable_parameters(self):
        """
        Return generator of regularizable parameters. Right now, all non-flat
        vectors are assumed to be regularizabled, presumably because only
        biases are flat.

        :return:
        """
        for param in self.parameters():
            if len(param.size()) > 1:
                yield param

    def eval_np(self, *args, **kwargs):
        """
        Eval this module with a numpy interface

        Same as a call to __call__ except all Variable input/outputs are
        replaced with numpy equivalents.

        Assumes the output is either a single object or a tuple of objects.
        """
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        outputs = self.__call__(*torch_args, **torch_kwargs)

        if isinstance(outputs, tuple) or isinstance(outputs, list):
            # return tuple(np_ify(x) for x in outputs)            # return tuple(np_ify(x) for x in outputs)
            return recursive_np_ify(outputs)
        else:
            return np_ify(outputs)


def torch_ify(np_array_or_other):
    if isinstance(np_array_or_other, np.ndarray):
        return ptu.from_numpy(np_array_or_other)
    else:
        return np_array_or_other


def np_ify(tensor_or_other):
    if isinstance(tensor_or_other, Variable):
        return ptu.get_numpy(tensor_or_other)
    else:
        return tensor_or_other


def recursive_np_ify(object_holding_tensor):
    if isinstance(object_holding_tensor, torch.Tensor):
        return np_ify(object_holding_tensor)
    elif isinstance(object_holding_tensor, dict):
        return {k: np_ify(v) for k, v in object_holding_tensor.items()}
    elif isinstance(object_holding_tensor, tuple) or isinstance(object_holding_tensor, list):
        return tuple([recursive_np_ify(el) for el in object_holding_tensor])
    elif isinstance(object_holding_tensor, np.ndarray):
        return object_holding_tensor

def rgetattr(obj, attr, *args):
    """See https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects"""
    def _getattr(obj, attr):
        if obj is None:
            return None
        return getattr(obj, attr, *args)
    return reduce(_getattr, [obj] + attr.split('.'))


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)