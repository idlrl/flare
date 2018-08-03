import numpy as np
import torch
from torch.distributions.distribution import Distribution


class Deterministic(Distribution):
    """
    Creates a deterministic distribution parameterized by `vector`
    """

    def __init__(self, vector, validate_args=None):
        self._param = vector
        batch_shape = self._param.size()[:-1] \
                      if self._param.ndimension() > 1 else torch.Size()
        super(Deterministic, self).__init__(
            batch_shape, validate_args=validate_args)

    @property
    def param_shape(self):
        return self._param.size()

    @property
    def mean(self):
        return self._param

    def sample(self):
        return self._param

    def log_prob(self, a):
        return torch.zeros(a.shape)
