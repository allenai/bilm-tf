
import unittest
import os
import json

import numpy as np
import tensorflow as tf

from bilm.model import BidirectionalLanguageModel
from bilm.data import Batcher
from bilm.elmo import weight_layers

FIXTURES = 'tests/fixtures/model/'



class TestWeightedLayers(unittest.TestCase):
    def setUp(self):
        self.sess = tf.Session()

    def tearDown(self):
        tf.reset_default_graph()
        self.sess.close()

    def test_weighted_layers(self):
        # create the Batcher
        vocab_file = os.path.join(FIXTURES, 'vocab_test.txt')
        batcher = Batcher(vocab_file, 50)

        # load the model
        options_file = os.path.join(FIXTURES, 'options.json')
        weight_file = os.path.join(FIXTURES, 'lm_weights.hdf5')
        character_ids = tf.placeholder('int32', (None, None, 50))
        model = BidirectionalLanguageModel(
            options_file, weight_file, character_ids, 4)

        weighted_ops = weight_layers(2, model, 1.0)

        # initialize
        self.sess.run(tf.global_variables_initializer())

        # should have two trainable variables per weighted layer
        self.assertEqual(len(tf.trainable_variables()), 4)
        # and one regularizer per weighted layer
        self.assertEqual(
            len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            2
        )

        # Set the variables.
        weights = [[np.array([0.1, 0.3, 0.5]), np.array([1.1])],
                   [np.array([0.2, 0.4, 0.6]), np.array([0.88])]]
        for k in range(2):
            with tf.variable_scope('', reuse=True):
                W = tf.get_variable('ELMo_W_{}'.format(k))
                gamma = tf.get_variable('ELMo_gamma_{}'.format(k))
                _ = self.sess.run(
                    [W.assign(weights[k][0]), gamma.assign(weights[k][1])]
                )

        # make some data
        sentences = [
            ['The', 'first', 'sentence', '.'],
            ['The', 'second'],
            ['Third']
        ]
        X_chars = batcher.batch_sentences(sentences)

        ops = model.get_ops()
        lm_embeddings, mask, weighted0, weighted1 = self.sess.run(
            [ops['lm_embeddings'], ops['mask'],
             weighted_ops[0][0], weighted_ops[1][0]],
            feed_dict={character_ids: X_chars}
        )
        actual_elmo = [weighted0, weighted1]

        # check the mask first
        expected_mask = [[True, True, True, True],
                         [True, True, False, False],
                         [True, False, False, False]]
        self.assertTrue((expected_mask == mask).all())

        # Now compute the actual weighted layers
        for k in range(2):
            normed_weights = np.exp(weights[k][0]) / np.sum(
                                                np.exp(weights[k][0]))
            # masked layer normalization
            expected_elmo = np.zeros((3, 4, lm_embeddings.shape[-1]))
            for j in range(3):  # number of LM layers
                mean = np.mean(lm_embeddings[:, j, :, :][mask])
                std = np.std(lm_embeddings[:, j, :, :][mask])
                normed_lm_embed = (lm_embeddings[:, j, :, :] - mean) / (
                    std + 1E-12)
                expected_elmo += normed_weights[j] * normed_lm_embed

            # the scale parameter
            expected_elmo *= weights[k][1]
            self.assertTrue(
                np.allclose(expected_elmo, actual_elmo[k], atol=1e-6)
            )


if __name__ == '__main__':
    unittest.main()


