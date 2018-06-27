import torch
import numpy as np


def make_hierarchy_of_tensors(data, dtype, device, shape):
    """
    Given a (nested) list of arrays (`data`), each with a `shape`,
    return a (nested) list of torch tensors such that only the leaf nodes that
    all should match `shape` will be converted. The other intermediate nodes
    keep the same (still list).
    """
    assert hasattr(data, "__len__"), "no leaf with the shape is found!"
    try:
        np_array = np.array(data)
        if len(np_array.shape) == len(shape) + 1:  ## plus batch size
            if list(np_array.shape[1:]) == list(shape):
                ## we have reached a leaf node
                return torch.from_numpy(np_array.astype(dtype)).to(device)
    except:
        ## conversion fails
        ## recursively traverse down the hierarchy
        pass

    return [make_hierarchy_of_tensors(d, dtype, device, shape) for d in data]


def batch_size(b):
    """
    Obtain the length of a batch. The batch can be a tensor or a list.
    """
    return (b.size()[0] if isinstance(b, torch.Tensor) else len(b))


def transpose(hier_tensors):
    """
    Given a hierarchy of tensors, at the highest level we transpose the structure.
    Namely, we make a frame by taking out the i-th child of each node at the current level.
    """
    assert isinstance(hier_tensors,
                      list) and hier_tensors, "must be a non-empty list!"
    leaf_child = isinstance(hier_tensors[0], torch.Tensor)
    ### check all the children's type
    for node in hier_tensors:
        assert isinstance(node, torch.Tensor) == leaf_child, \
            "all children must have the same type!"

    ## we require that the children at each level are sorted in the
    ## descending order of their lengths
    seq_lens = [batch_size(node) for node in hier_tensors]
    assert seq_lens == sorted(seq_lens, reverse=True), \
        "please sort the sequences in the descending order of sequence lengths!"

    max_len = seq_lens[0]
    transposed = []
    for i in range(max_len):
        if leaf_child:
            new_node = torch.stack(
                [t[i] for t in hier_tensors if i < t.size()[0]])
        else:
            new_node = [node[i] for node in hier_tensors if i < len(node)]
        transposed.append(new_node)
    return transposed


def recurrent_group(seq_inputs, insts, init_states, step_func):
    """
    Strip a sequence level and apply the stripped inputs to `step_func`
    provided by the user.

    seq_inputs: collection of sequences, each being either a tensor or a list
    insts: collection of static instances, each being a tensor
    init_states: collection of initial states, each being a tensor
    """
    assert isinstance(seq_inputs, list)
    assert isinstance(insts, list)
    assert isinstance(init_states, list)
    for ipt in seq_inputs:
        assert isinstance(ipt, list), \
            "Each sequential input should be a collection (batch) of sequences!"

    ## We might have multiple sequential inputs.
    ## (For example,
    ##   the first sequential input is a collection of temporal sequences of sentences,
    ##   the second sequential input is a collection of temporal sequences of images, and
    ##   each sentence is paired with an image.)
    ## In this scenario, after the transpose the numbers of frames should be
    ## the same.
    ## If two sequential inputs cannot satisfy this requirement, then consider separating
    ## them by calling recurrent_group individually, because there cannot be any interaction
    ## between them at every time step.

    ## The following block checks this requirement.
    def check_num_sequences_equal(seq_inputs):
        assert len(set([len(si) for si in seq_inputs])) == 1, \
            "Each sequential input should have the same number of sequences!"

    def check_sequences_aligned(seq_inputs):
        len_mat = [map(batch_size, si) for si in seq_inputs]
        trans_len_mat = zip(*len_mat)
        assert all([len(set(r)) == 1 for r in trans_len_mat]), \
            "Some sequences are not temporally aligned!"

    check_num_sequences_equal(seq_inputs)
    check_sequences_aligned(seq_inputs)

    for inst in insts:
        assert isinstance(inst, torch.Tensor)
        assert len(seq_inputs[0]) == inst.size()[0], \
            "The number of static instances should be the same with the number of sequences!"
    for init_state in init_states:
        assert isinstance(init_state, torch.Tensor)
        assert len(seq_inputs[0]) == init_state.size()[0], \
            "The number of initial states should be the same with the number of sequences!"

    ## we need to sort the sequential inputs by the seq lengths
    seq_lens = [(i, batch_size(seq)) for i, seq in enumerate(seq_inputs[0])]
    seq_lens = sorted(seq_lens, key=lambda p: p[1], reverse=True)
    sorted_idx = [l[0] for l in seq_lens]
    seq_inputs = [[ipt[i] for i in sorted_idx] for ipt in seq_inputs]
    insts = [inst[sorted_idx] for inst in insts]
    init_states = [init_state[sorted_idx] for init_state in init_states]

    ## call the step function frame by frame
    frames_n = seq_lens[0][1]
    seq_frames = []
    for seq in seq_inputs:
        seq_frames.append(transpose(seq))
    states = init_states
    transposed_out_frames = []
    for i in range(frames_n):
        in_frames = []
        frame_size = None
        for frs in seq_frames:
            size = batch_size(frs[i])
            if frame_size is None:
                frame_size = size
            else:
                assert frame_size == size, "all the input frames must have the same size!!"
            in_frames.append(frs[i])

        ## we should cut the insts and states by the frame_size
        states = [s[:frame_size] for s in states]
        insts = [i[:frame_size] for i in insts]
        out_frames, states = step_func(*(in_frames + insts + states))
        #        out_frames += states
        transposed_out_frames.append(out_frames)

    seq_outs = [
        transpose(list(frames)) for frames in zip(*transposed_out_frames)
    ]

    ## we have to recover the order of sequences using `sorted_idx`
    reverse_idx = zip(*sorted(
        list(enumerate(sorted_idx)), key=lambda p: p[1]))[0]
    seq_outs = [[out[i] for i in reverse_idx] for out in seq_outs]

    return seq_outs
