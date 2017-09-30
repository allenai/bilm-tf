
import unittest
import os
import json
import h5py

import numpy as np
import tensorflow as tf

from bilm.model import BidirectionalLanguageModel
from bilm.data import Batcher

FIXTURES = 'tests/fixtures/model/'


class TestBidirectionalLanguageModel(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()

    def tearDown(self):
        self.sess.close()

    def test_bilm(self):
        # get the raw data
        with open(os.path.join(FIXTURES,
                               'lm_embeddings_sentences.json')) as fin:
            sentences = json.load(fin)
    
        # the expected embeddings
        expected_lm_embeddings = []
        for k in range(len(sentences)):
            embed_fname = os.path.join(
                FIXTURES, 'lm_embeddings_{}.hdf5'.format(k)
            )
            expected_lm_embeddings.append([])
            with h5py.File(embed_fname, 'r') as fin:
                for i in range(10):
                    sent_embeds = fin['%s' % i][...]
                    sent_embeds_concat = np.concatenate(
                        (sent_embeds[0, :, :], sent_embeds[1, :, :]),
                        axis=-1
                    )
                    expected_lm_embeddings[-1].append(sent_embeds_concat)

        # create the Batcher
        vocab_file = os.path.join(FIXTURES, 'vocab_test.txt')
        batcher = Batcher(vocab_file, 50)

        # load the model
        options_file = os.path.join(FIXTURES, 'options.json')
        weight_file = os.path.join(FIXTURES, 'lm_weights.hdf5')
        character_ids = tf.placeholder('int32', (None, None, 50))
        model = BidirectionalLanguageModel(
            options_file, weight_file, character_ids, 4)

        # get the ops to compute embeddings
        ops = model.get_ops()

        # initialize
        self.sess.run(tf.global_variables_initializer())

        # We shouldn't have any trainable variables
        self.assertEqual(len(tf.trainable_variables()), 0)

        # will run 10 batches of 3 sentences
        for i in range(10):
            # make a batch of sentences
            batch_sentences = []
            for k in range(3):
                sentence = sentences[k][i].strip().split()
                batch_sentences.append(sentence)

            X = batcher.batch_sentences(batch_sentences)
            lm_embeddings, lengths = self.sess.run(
                [ops['lm_embeddings'], ops['lengths']],
                feed_dict={character_ids: X}
            )

            actual_lengths = [len(sent) for sent in batch_sentences]
            self.assertEqual(actual_lengths, list(lengths))

            # get the expected embeddings and compare!
            expected_y = [expected_lm_embeddings[k][i] for k in range(3)]
            for k in range(3):
                self.assertTrue(
                    np.allclose(
                        lm_embeddings[k, 2, :lengths[k], :],
                        expected_y[k],
                        atol=1.0e-6
                    )
                )

        # Finally, check that the states are being updated properly.
        # All batches were size=3, so last element of states should always
        # be zero.
        third_states = []
        for direction in ['forward', 'backward']:
            states = self.sess.run(model._lm_graph.lstm_init_states[direction])
            for i in range(2):
                for state in states[i]:
                    self.assertTrue(np.sum(np.abs(state[-1, :])) < 1e-7)
                    third_states.append(state[2, :])

        # Run a batch with size=2, the third state should not have been updated
        _ = self.sess.run(
            ops['lm_embeddings'],
            feed_dict={character_ids: np.ones((2, 5, 50), dtype=np.int32)}
        )
        k = 0
        for direction in ['forward', 'backward']:
            states = self.sess.run(model._lm_graph.lstm_init_states[direction])
            for i in range(2):
                for state in states[i]:
                    self.assertTrue(
                        np.allclose(third_states[k], state[2, :], atol=1e-6)
                    )
                    k += 1


if __name__ == '__main__':
    unittest.main()

