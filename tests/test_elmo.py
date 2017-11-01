
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
    def tearDown(self):
        tf.reset_default_graph()
        self.sess.close()

    def setUp(self):
        self.sess = tf.Session()

    def _check_weighted_layer(self, l2_coef, do_layer_norm, use_top_only):
        # create the Batcher
        vocab_file = os.path.join(FIXTURES, 'vocab_test.txt')
        batcher = Batcher(vocab_file, 50)

        # load the model
        options_file = os.path.join(FIXTURES, 'options.json')
        weight_file = os.path.join(FIXTURES, 'lm_weights.hdf5')
        character_ids = tf.placeholder('int32', (None, None, 50))
        model = BidirectionalLanguageModel(
            options_file, weight_file, max_batch_size=4)
        bilm_ops = model(character_ids)

        weighted_ops = []
        for k in range(2):
            ops = weight_layers(str(k), bilm_ops, l2_coef=l2_coef, 
                                     do_layer_norm=do_layer_norm,
                                     use_top_only=use_top_only)
            weighted_ops.append(ops)

        # initialize
        self.sess.run(tf.global_variables_initializer())

        n_expected_trainable_weights = 2 * (1 + int(not use_top_only))
        self.assertEqual(len(tf.trainable_variables()),
                         n_expected_trainable_weights)
        # and one regularizer per weighted layer
        n_expected_reg_losses = 2 * int(not use_top_only)
        self.assertEqual(
            len(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            n_expected_reg_losses,
        )

        # Set the variables.
        weights = [[np.array([0.1, 0.3, 0.5]), np.array([1.1])],
                   [np.array([0.2, 0.4, 0.6]), np.array([0.88])]]
        for k in range(2):
            with tf.variable_scope('', reuse=True):
                if not use_top_only:
                    W = tf.get_variable('{}_ELMo_W'.format(k))
                    _ = self.sess.run([W.assign(weights[k][0])])
                gamma = tf.get_variable('{}_ELMo_gamma'.format(k))
                _ = self.sess.run([gamma.assign(weights[k][1])])

        # make some data
        sentences = [
            ['The', 'first', 'sentence', '.'],
            ['The', 'second'],
            ['Third']
        ]
        X_chars = batcher.batch_sentences(sentences)

        ops = model(character_ids)
        lm_embeddings, mask, weighted0, weighted1 = self.sess.run(
            [ops['lm_embeddings'], ops['mask'],
             weighted_ops[0]['weighted_op'], weighted_ops[1]['weighted_op']],
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
            normed_weights = np.exp(weights[k][0] + 1.0 / 3) / np.sum(
                                  np.exp(weights[k][0] + 1.0 / 3))
            # masked layer normalization
            expected_elmo = np.zeros((3, 4, lm_embeddings.shape[-1]))
            if not use_top_only:
                for j in range(3):  # number of LM layers
                    if do_layer_norm:
                        mean = np.mean(lm_embeddings[:, j, :, :][mask])
                        std = np.std(lm_embeddings[:, j, :, :][mask])
                        normed_lm_embed = (lm_embeddings[:, j, :, :] - mean) / (
                            std + 1E-12)
                        expected_elmo += normed_weights[j] * normed_lm_embed
                    else:
                        expected_elmo += normed_weights[j] * lm_embeddings[
                                                                    :, j, :, :]
            else:
                expected_elmo += lm_embeddings[:, -1, :, :]

            # the scale parameter
            expected_elmo *= weights[k][1]
            self.assertTrue(
                np.allclose(expected_elmo, actual_elmo[k], atol=1e-6)
            )

    def test_weighted_layers(self):
        self._check_weighted_layer(1.0, do_layer_norm=True, use_top_only=False)

    def test_weighted_layers_no_norm(self):
        self._check_weighted_layer(1.0, do_layer_norm=False, use_top_only=False)

    def test_weighted_layers_top_only(self):
        self._check_weighted_layer(None, do_layer_norm=False, use_top_only=True)


if __name__ == '__main__':
    unittest.main()


