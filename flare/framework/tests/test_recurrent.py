import flare.framework.recurrent as rc
import unittest
import numpy as np
import torch


def tensor_lists_equal(t1, t2):
    """
    Given two (nested) lists of tensors, return whether they are equal.
    """
    if isinstance(t1, torch.Tensor) and isinstance(t2, torch.Tensor):
        ## round in case of floating errors
        t1 = np.round(t1.data.numpy(), decimals=5)
        t2 = np.round(t2.data.numpy(), decimals=5)
        return np.array_equal(t1, t2)

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
    def test_multi_sequential_inputs(self):
        ## we first make a batch of temporal sequences of sentences
        ## Let's suppose that each word has a 1d embedding
        sentences = [## temporal sequence 1
                     [[[0.3], [0.4], [0.5]],          ## sentence 1
                      [[0.1], [0.2]]],                ## sentence 2
                     ## temporal sequence 2
                     [[[0.3], [0.4], [0.5]],          ## sentence 3
                      [[0.2], [0.2]],                 ## sentence 4
                      [[1.0], [0.2], [0.4], [0.5]]],  ## sentence 5
        ]
        sentence_tensors = rc.make_hierarchy_of_tensors(sentences, "float32",
                                                        "cpu", [1])

        ## we then make a batch of temporal sequences of images
        images = [## temporal sequence 1
                  [[0.1, 0.3],                 ## image 1
                   [0, -1]],                   ## image 2
                  ## temporal sequence 2
                  [[0, 1],                     ## image 3
                   [1, 0],                     ## image 4
                   [1, 1]],                    ## image 5
        ]
        image_tensors = rc.make_hierarchy_of_tensors(images, "float32", "cpu",
                                                     [2])

        states = [
            [-2, -4, -6, -8],  ## temporal sequence 1
            [-1, -2, -3, -4],  ## temporal sequence 2
        ]
        state_tensors = rc.make_hierarchy_of_tensors(states, "float32", "cpu",
                                                     [4])

        def step_func(sentence, image, state):
            """
            We compute the first output by doing outer product between the
            average word embedding and the image embedding.
            We compute the second output by adding the average word embedding
            and the image embedding.
            We directly add the state mean value to the output
            and update the state by multiplying it with -1.
            """
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
                        torch.tensor([[-4.9600, -4.8800], [5.0000, 4.8500]]),
                        torch.tensor([[-2.5000, -2.1000], [2.7000, 2.5000],
                                      [-1.9750, -1.9750]])
                    ],  ## out1
                    [
                        torch.tensor([[-4.4000], [4.6500]]), torch.tensor(
                            [[-1.6000], [3.2000], [-0.9750]])
                    ],  ## out2
                    [
                        torch.tensor([[2., 4., 6., 8.], [-2., -4., -6., -8.]]),
                        torch.tensor([[1., 2., 3., 4.], [-1., -2., -3., -4.],
                                      [1., 2., 3., 4.]])
                    ]  ## state
                ]))

    def test_hierchical_sequences(self):
        sentences = [## temporal sequence 1
                     [[[0.3], [0.4], [0.5]],          ## sentence 1
                      [[0.1], [0.2]]],                ## sentence 2
                     ## temporal sequence 2
                     [[[0.3], [0.4], [0.5]],          ## sentence 3
                      [[0.2], [0.2]],                 ## sentence 4
                      [[1.0], [0.2], [0.4], [0.5]]],  ## sentence 5
        ]
        sentence_tensors = rc.make_hierarchy_of_tensors(sentences, "float32",
                                                        "cpu", [1])

        sentence_states = [
            [-2, -4, -6, -8],  ## temporal sequence 1
            [-1, -2, -3, -4],  ## temporal sequence 2
        ]
        sentence_state_tensors = rc.make_hierarchy_of_tensors(
            sentence_states, "float32", "cpu", [4])

        word_states = [
            [1.0, 1.0],  ## sentence 1
            [-1.0, -1.0],  ## sentence 3
        ]
        word_state_tensors = rc.make_hierarchy_of_tensors(
            word_states, "float32", "cpu", [2])

        ## This hierarchical function does the following things:
        ## 1. For each word in each sentence, we add the word state
        ##    to the word embedding, and the word state keeps the same all the time
        ## 2. We take the last output of the words and the word states
        ## 3. In the higher level, we multiply the last word output with the sentence state,
        ##    and update the sentence state by multiplying it with -1
        def step_func(sentence, sentence_state, word_state):
            assert isinstance(sentence, list)

            def inner_step_func(w, ws):
                ### w is the current word emebdding
                ### ws is the current word state
                assert isinstance(w, torch.Tensor)
                assert isinstance(ws, torch.Tensor)
                ## return output and updated state
                return [w + ws.mean(-1).unsqueeze(-1)], [ws]

            outputs, word_states = rc.recurrent_group(
                seq_inputs=[sentence],
                insts=[],
                init_states=[word_state],
                step_func=inner_step_func)

            last_outputs = torch.stack([o[-1] for o in outputs])
            last_word_states = torch.stack([s[-1] for s in word_states])
            ## we compute the output by multipying the sentence state
            ## with the last word state
            out = last_outputs * sentence_state
            return [out], [sentence_state * -1, last_word_states]

        outs, sentence_states, word_states \
            = rc.recurrent_group(seq_inputs=[sentence_tensors],
                                 insts=[],
                                 init_states=[sentence_state_tensors,
                                              word_state_tensors],
                                 step_func=step_func)
        self.assertTrue(
            tensor_lists_equal(outs, [
                torch.tensor([[-3.0, -6.0, -9.0, -12.0], [2.4, 4.8, 7.2, 9.6]
                              ]), torch.tensor([[0.5, 1.0, 1.5, 2.0], [
                                  -0.8, -1.6, -2.4, -3.2
                              ], [0.5, 1.0, 1.5, 2.0]])
            ]))
        self.assertTrue(
            tensor_lists_equal(sentence_states, [
                torch.tensor([[2., 4., 6., 8.], [-2., -4., -6., -8.]]),
                torch.tensor([[1., 2., 3., 4.], [-1., -2., -3., -4.],
                              [1., 2., 3., 4.]])
            ]))
        self.assertTrue(
            tensor_lists_equal(word_states, [
                torch.tensor([[1., 1.], [1., 1.]]), torch.tensor(
                    [[-1., -1.], [-1., -1.], [-1., -1.]])
            ]))


if __name__ == "__main__":
    unittest.main()
