

import unittest
import os
import shutil
import tempfile

import tensorflow as tf
import numpy as np

from bilm.training import train, test, load_vocab, \
                                load_options_latest_checkpoint
from bilm.data import LMDataset, BidirectionalLMDataset

FIXTURES = 'tests/fixtures/train/'

class TestLanguageModel(unittest.TestCase):
    def tearDown(self):
        shutil.rmtree(self.tmp_dir)
        os.remove(self.tmp_file)
        tf.reset_default_graph()

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        (_, self.tmp_file) = tempfile.mkstemp()

    def _get_data(self, bidirectional, use_chars, test=False):
        vocab_file = os.path.join(FIXTURES, 'vocab.txt')
        if use_chars:
            vocab = load_vocab(vocab_file, 10)
        else:
            vocab = load_vocab(vocab_file, None)

        prefix = os.path.join(FIXTURES, 'data.txt')

        if bidirectional:
            data = BidirectionalLMDataset(prefix, vocab, test=test)
        else:
            data = LMDataset(prefix, vocab, test=test, reverse=False)

        return data, vocab

    def _get_vocab_data_options(self, bidirectional, use_chars,
                                share_embedding_softmax=False):
        data, vocab = self._get_data(bidirectional, use_chars)
        options = {
            'n_tokens_vocab': vocab.size,
            'n_negative_samples_batch': 16,
            'n_train_tokens': 134,
            'batch_size': 2,
            'unroll_steps': 10,
            'n_epochs': 50,
            'all_clip_norm_val': 1.0,
            'dropout': 0.1,
            'lstm': {'dim': 16, 'projection_dim': 8, 'n_layers': 2},
            'bidirectional': bidirectional,
        }

        if use_chars:
            options['char_cnn'] = {
               'n_characters': 261,
               'max_characters_per_token': 10,
               'filters': [
                    [1, 8],
                    [2, 8],
                    [3, 16],
                    [4, 32],
                    [5, 64],
                ],
                'activation': 'tanh',
                'embedding': {'dim': 4},
                'n_highway': 1,
            }

        if share_embedding_softmax:
            options['share_embedding_softmax'] = True

        return vocab, data, options

    def test_train_single_direction(self):
        vocab, data, options = self._get_vocab_data_options(False, False)
        train(options, data, 1, self.tmp_dir, self.tmp_dir)

        # now test
        tf.reset_default_graph()
        options, ckpt_file = load_options_latest_checkpoint(self.tmp_dir)
        data_test, vocab_test = self._get_data(False, False, True)
        perplexity = test(options, ckpt_file, data_test, batch_size=1)
        self.assertTrue(perplexity < 20.0)

    def test_train_bilm_chars(self):
        vocab, data, options = self._get_vocab_data_options(True, True)
        train(options, data, 1, self.tmp_dir, self.tmp_dir)

        # now test
        tf.reset_default_graph()
        options, ckpt_file = load_options_latest_checkpoint(self.tmp_dir)
        data_test, vocab_test = self._get_data(True, True, True)
        perplexity = test(options, ckpt_file, data_test, batch_size=1)
        self.assertTrue(perplexity < 20.0)

    def test_shared_variables(self):
        vocab, data, options = self._get_vocab_data_options(True, True)
        options['n_epochs'] = 1
        train(options, data, 2, self.tmp_dir, self.tmp_dir)
        self.assertEqual(len(tf.global_variables()), 64)

    def test_train_shared_softmax_embedding(self):
        bidirectional = True
        use_chars = False

        vocab, data, options = self._get_vocab_data_options(
            bidirectional, use_chars, share_embedding_softmax=True)
        train(options, data, 1, self.tmp_dir, self.tmp_dir)

        # now test
        tf.reset_default_graph()
        options, ckpt_file = load_options_latest_checkpoint(self.tmp_dir)
        data_test, vocab_test = self._get_data(
            bidirectional, use_chars, test=True)
        perplexity = test(options, ckpt_file, data_test, batch_size=1)
        self.assertTrue(perplexity < 20.0)

    def test_train_shared_softmax_no_chars(self):
        bidirectional = True
        use_chars = True
        vocab, data, options = self._get_vocab_data_options(
            bidirectional, use_chars, share_embedding_softmax=True)
        # character inputs and sharing weights not suppported
        with self.assertRaises(ValueError):
            train(options, data, 1, self.tmp_dir, self.tmp_dir)

    def test_train_skip_connections(self):
        bidirectional = True
        use_chars = False
        vocab, data, options = self._get_vocab_data_options(
            bidirectional, use_chars)
        options['lstm']['use_skip_connections'] = True
        train(options, data, 1, self.tmp_dir, self.tmp_dir)

        # now test
        tf.reset_default_graph()
        options, ckpt_file = load_options_latest_checkpoint(self.tmp_dir)
        data_test, vocab_test = self._get_data(
            bidirectional, use_chars, test=True)
        perplexity = test(options, ckpt_file, data_test, batch_size=1)
        self.assertTrue(perplexity < 20.0)


if __name__ == '__main__':
    unittest.main()

