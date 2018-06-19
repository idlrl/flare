import torch
import torch.nn as nn
from torch.distributions import Categorical


def q_categorical(q_value):
    """
    Generate a Categorical distribution object given a Q value.
    We construct a one-hot distribution according to the Q value.
    This is for prediction only and we do not require gradients.
    """
    assert len(q_value.size()) == 2, "[batch_size, num_actions]"
    _, max_id = q_value.max(-1)  ## max along the second dimension
    ### max_id is 1D
    prob = one_hot(max_id, depth=q_value.size()[-1])
    return Categorical(probs=prob)


def inner_prod(x, y):
    """
    Get the inner product of two vectors
    """
    return torch.sum(torch.mul(x, y), dim=1).view(-1, 1)


def idx_select(input, idx):
    """
    Given an input vector (Tensor) and an idx (int or IntTensor),
    select the entry of the vector according to the idx.
    """
    assert len(input.size()) == 2

    if isinstance(idx, int):
        return input[:, idx].view(-1, 1)
    else:
        assert isinstance(idx, torch.Tensor)
        assert len(idx.size()) == 2, "idx should be two-dimensional!"
        idx_ = idx.int()
        ## This might be able to be done with .gather()
        ## However, currently there is grad issue with .gather()
        return inner_prod(input, one_hot(idx.squeeze(-1), input.size()[-1]))


def one_hot(idx, depth):
    ### idx will be just one-dimensional
    ### the embedded vectors will be two-dimensional
    ones = torch.eye(depth)
    if idx.is_cuda:
        ones = ones.to(idx.get_device())
    return ones.index_select(0, idx.long())
