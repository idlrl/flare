from copy import deepcopy
import numpy as np


def split_list(l, sizes):
    """
    Split a list into several chunks, each chunk with a size in `sizes`.
    """
    chunks = []
    offset = 0
    for size in sizes:
        chunks.append(l[offset:offset + size])
        offset += size
    return chunks


def concat_dicts(dict_list):
    """
    Concatenate values of each key from a list of dictionary.

    The type of values should be `numpy.ndarray`, and the result is
    the concatenation of these values at axis=0.
    """
    D = {}
    for d in dict_list:
        if not D:
            D = deepcopy(d)
        else:
            assert (set(d.keys()) == set(D.keys()))
            for k in D:
                assert isinstance(d[k], type(D[k]))
                if type(d[k]) == list:
                    D[k] += d[k]
                elif type(d[k] == np.ndarray):
                    D[k] = np.concatenate([D[k], d[k]])
                else:
                    raise TypeError("only numpy.ndarray or list is accepted")

    return D


def split_dict(D, starts):
    """
    Inverse operation of `concat_dicts`.
    """
    ret = []
    for i in range(len(starts) - 1):
        d = {}
        for k, v in D.items():
            d[k] = deepcopy(v[starts[i]:starts[i + 1]])
        ret.append(d)
    return ret
