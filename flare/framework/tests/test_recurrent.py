import flare.framework.recurrent as rc
import unittest
import numpy as np
import torch


def tensor_lists_equal(t1, t2):
    """
    Given two (nested) lists of tensors, return whether they are equal.
    """
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        return bool((t1 == t2).all().item())

    assert isinstance(t1, list)
    assert isinstance(t2, list)
    assert len(t1) == len(t2)

    for t1_, t2_ in zip(t1, t2):
        if not tensor_lists_equal(t1_, t2_):
            return False
    return True


class TestHierarchyTensorTranspose(unittest.TestCase):
    def test_transpose(self):
        data = [[[[3], [4], [5]], [[1], [2]], [[10], [2], [4], [5]]],
                [[[3], [4], [5]], [[1], [2]]]]

        ht = rc.make_hierarchy_of_tensors(data, "int64", "cpu", (1, ))
        ht_ = rc.transpose(rc.transpose(ht))  ## this should convert back to ht
        self.assertTrue(tensor_lists_equal(ht, ht_))


class TestRecurrentGroup(unittest.TestCase):
    def test_recurrent_group(self):
        ## we first make a batch of temporal sequences of sentences
        ## Let's suppose that each word has a 1d embedding
        sentences = [## temporal sequence 1
                     [[[0.3], [0.4], [0.5]],          ## sentence 1
                      [[0.2], [0.2]],                 ## sentence 2
                      [[1.0], [0.2], [0.4], [0.5]]],  ## sentence 3
                     ## temporal sequence 2
                     [[[0.3], [0.4], [0.5]],          ## sentence 4
                      [[0.1], [0.2]]]                 ## sentence 5
        ]
        sentence_tensors = rc.make_hierarchy_of_tensors(sentences, "float32",
                                                        "cpu", [1])

        ## we then make a batch of temporal sequences of images
        images = [## temporal sequence 1
                  [[0, 1],                     ## image 1
                   [1, 0],                     ## image 2
                   [1, 1]],                    ## image 3
                  ## temporal sequence 2
                  [[0.1, 0.3],                 ## image 4
                   [0, -1]]                    ## image 5
        ]
        image_tensors = rc.make_hierarchy_of_tensors(images, "float32", "cpu",
                                                     [2])

        states = [
            [-1, -2, -3, -4],  ## temporal sequence 1
            [-2, -4, -6, -8]  ## temporal sequence 2
        ]
        state_tensors = rc.make_hierarchy_of_tensors(states, "float32", "cpu",
                                                     [4])

        def step_func(sentence, image, state):
            """
            We have no `insts` and `states`
            """
            ## We compute the first output by doing outer product between the
            ## average word embedding and the image embedding.
            ## We compute the second output by adding the average word embedding
            ## and the average image embedding.
            ## We directly add the state mean value to the output
            ## and update the state by multiplying it with -1.
            assert isinstance(sentence, list)
            assert isinstance(image, torch.Tensor)
            assert isinstance(state, torch.Tensor)
            sentence = torch.stack([sen.mean(0) for sen in sentence])
            assert sentence.size()[0] == image.size()[0]

            mean_state = state.mean(-1).unsqueeze(-1)
            out1 = torch.bmm(sentence.unsqueeze(2), image.unsqueeze(1)).view(
                sentence.size()[0], -1) + mean_state
            out2 = sentence + image.mean(-1).unsqueeze(-1) + mean_state
            return [out1, out2], [state * -1]

        outs = rc.recurrent_group([sentence_tensors, image_tensors], [],
                                  [state_tensors], step_func)

        self.assertTrue(
            tensor_lists_equal(
                outs,
                [
                    [
                        torch.tensor([[-2.5000, -2.1000], [2.7000, 2.5000],
                                      [-1.9750, -1.9750]]),
                        torch.tensor([[-4.9600, -4.8800], [5.0000, 4.8500]])
                    ],  ## out1
                    [
                        torch.tensor([[-1.6000], [3.2000], [-0.9750]]),
                        torch.tensor([[-4.4000], [4.6500]])
                    ],  ## out2
                    [
                        torch.tensor([[1., 2., 3., 4.], [-1., -2., -3., -4.],
                                      [1., 2., 3., 4.]]),
                        torch.tensor([[2., 4., 6., 8.], [-2., -4., -6., -8.]])
                    ]  ## state
                ]))


if __name__ == "__main__":
    unittest.main()
