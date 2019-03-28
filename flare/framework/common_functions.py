from collections import Iterable
import torch
import torch.nn as nn
from torch.distributions import Categorical
import operator
from functools import reduce
import flare.framework.recurrent as rc


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


def get_avg_cost(costs):
    cost_total, weight_total = sum_cost_tensor(costs)
    avg_cost = cost_total / weight_total
    return avg_cost


def sum_cost_array(costs):
    if isinstance(costs, Iterable):
        costs, ns = zip(*map(sum_cost_array, costs))
        return sum(costs), sum(ns)
    else:
        return costs, 1


def sum_cost_tensor(costs):
    if isinstance(costs, torch.Tensor):
        return (costs.view(-1).sum(), reduce(operator.mul, costs.size()))
    assert isinstance(costs, list)
    costs, ns = zip(*map(sum_cost_tensor, costs))
    return sum(costs), sum(ns)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class GRUCellReLU(nn.Module):
    """
    A self-implemented GRUCell with ReLU activation support
    """

    def __init__(self, input_size, hidden_size):
        super(GRUCellReLU, self).__init__()
        self.r_fc = nn.Linear(input_size + hidden_size, hidden_size)
        self.z_fc = nn.Linear(input_size + hidden_size, hidden_size)
        self.in_fc = nn.Linear(input_size, hidden_size)
        self.hn_fc = nn.Linear(hidden_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, input, hx=None):
        """
        Formulas from https://en.wikipedia.org/wiki/Gated_recurrent_unit
        """
        # if hx is None:
        #     hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        r = torch.sigmoid(self.r_fc(torch.cat((input, hx), dim=-1)))
        z = torch.sigmoid(self.z_fc(torch.cat((input, hx), dim=-1)))
        n = torch.relu(self.in_fc(input) + self.hn_fc(r * hx))
        return (1 - z) * n + z * hx


class BoW(nn.Module):
    """
    Convert a sentence to a compact BoW embedding.
    If one_more_hidden is True, then we add another hidden layer
    after the BoW embedding.

    dict_size: the vocabulary size
    dim:       the embedding/hidden size
    """

    def __init__(self, dict_size, dim, std=None, one_more_hidden=True):
        super(BoW, self).__init__()
        self.dim = dim
        self.embedding_table = nn.Embedding(dict_size, dim)
        if std is not None:
            assert std > 0
            self.embedding_table.weight.data.normal_(0, std)
        self.one_more_hidden = one_more_hidden
        if one_more_hidden:
            self.hidden = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())

    def forward(self, sentence):
        def embedding(word):
            return [self.embedding_table(word.squeeze(-1))], []

        embeddings, = rc.recurrent_group(
            seq_inputs=[sentence], step_func=embedding)
        bow = sequence_pooling(embeddings, pooling_type="sum")
        if self.one_more_hidden:
            return self.hidden(bow)
        return bow


def sequence_pooling(seq, pooling_type="average"):
    assert isinstance(seq, list), "seq must be a list!"
    for inst in seq:
        assert isinstance(inst, torch.Tensor), "seq must be the lowest level"
    if pooling_type == "average":
        f = torch.mean
    elif pooling_type == "sum":
        f = torch.sum
    elif pooling_type == "max":
        return torch.stack([torch.max(inst, dim=0)[0] for inst in seq])
    else:
        assert False, "Incorrect pooling_type!"
    return torch.stack([f(inst, dim=0) for inst in seq])


def sequence_last(seq):
    assert isinstance(seq, list), "seq must be a list!"
    for inst in seq:
        assert isinstance(inst, torch.Tensor), "seq must be the lowest level"
    return torch.stack([inst[-1] for inst in seq])


def prepare_ntd_reward(reward, discount_factor):
    assert isinstance(reward, list)
    ntd_reward = []
    for r_ in reward:
        assert isinstance(r_, torch.Tensor)
        r = r_.clone()
        assert len(r.size()) == 2, "r should be a 2d tensor!"
        for i in range(r.size()[0] - 2, -1, -1):
            r[i] += discount_factor * r[i + 1]
        ntd_reward.append(r)
    return ntd_reward


def prepare_ntd_value(value, discount_factor):
    assert isinstance(value, list)
    ntd_value = []
    for v_ in value:
        assert isinstance(v_, torch.Tensor)
        v = v_.clone()
        assert len(v.size()) == 2, "v should be a 2d tensor!"
        for i in range(v.size()[0] - 2, -1, -1):
            v[i] = discount_factor * v[i + 1]
        ntd_value.append(v)
    return ntd_value


def rl_safe_batchnorm(bn_cls):
    assert bn_cls == nn.BatchNorm1d \
        or bn_cls == nn.BatchNorm2d \
        or bn_cls == nn.BatchNorm3d

    class safeBatchNorm(bn_cls):
        """
        Reason for this customized batchnorm layer is that nn.BatchNorm
        requires the batch size > 1. Towards the end of training we might
        only have just one training sample from a single agent each time.
        """
        def __init__(self, dim):
            super().__init__(dim)

        def forward(self, x):
            # batch size might be 1 (one agent) towards training end
            if x.size()[0] == 1:
                return x
            return super().forward(x)

    return safeBatchNorm
