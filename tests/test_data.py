
import unittest
import tempfile
import os
import numpy as np

from bilm.data import UnicodeCharsVocabulary, Vocabulary, \
    Batcher, TokenBatcher, LMDataset, BidirectionalLMDataset

DATA_FIXTURES = 'tests/fixtures/data/'
TRAIN_FIXTURES = 'tests/fixtures/train/'


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        words = ['<S>', '</S>', '<UNK>', 'the', '.']
        (_, tmp) = tempfile.mkstemp()
        with open(tmp, 'w') as fout:
            fout.write('\n'.join(words))
        self._tmp = tmp

    def test_vocab_encode(self):
        vocab = Vocabulary(self._tmp)
        sentence = 'the unknown .'
        ids = vocab.encode(sentence)
        expected = np.array([0, 3, 2, 4, 1], dtype=np.int32)
        self.assertTrue((ids == expected).all())

    def test_vocab_encode_reverse(self):
        vocab = Vocabulary(self._tmp)
        sentence = '. unknown the'
        ids = vocab.encode(sentence, reverse=True)
        expected = np.array([1, 4, 2, 3, 0], dtype=np.int32)
        self.assertTrue((ids == expected).all())

    def tearDown(self):
        os.remove(self._tmp)

class TestUnicodeCharsVocabulary(unittest.TestCase):
    def setUp(self):
        words = ['the', '.', chr(256) + 't', '<S>', '</S>', '<UNK>']
        (_, tmp) = tempfile.mkstemp()
        with open(tmp, 'w') as fout:
            fout.write('\n'.join(words))
        self.vocab = UnicodeCharsVocabulary(tmp, 5)
        self._tmp = tmp

    def test_vocab_word_to_char_ids(self):
        char_ids = self.vocab.word_to_char_ids('th')
        expected = np.array([258, 116, 104, 259, 260], dtype=np.int32)
        self.assertTrue((char_ids == expected).all())

        char_ids = self.vocab.word_to_char_ids('thhhhh')
        expected = np.array([258, 116, 104, 104, 259])
        self.assertTrue((char_ids == expected).all())

        char_ids = self.vocab.word_to_char_ids(chr(256) + 't')
        expected = np.array([258, 196, 128, 116, 259], dtype=np.int32)
        self.assertTrue((char_ids == expected).all())

    def test_bos_eos(self):
        bos_ids = self.vocab.word_to_char_ids('<S>')
        self.assertTrue((bos_ids == self.vocab.bos_chars).all())

        bos_ids = self.vocab.word_char_ids[self.vocab.word_to_id('<S>')]
        self.assertTrue((bos_ids == self.vocab.bos_chars).all())

        eos_ids = self.vocab.word_to_char_ids('</S>')
        self.assertTrue((eos_ids == self.vocab.eos_chars).all())

        eos_ids = self.vocab.word_char_ids[self.vocab.word_to_id('</S>')]
        self.assertTrue((eos_ids == self.vocab.eos_chars).all())

    def test_vocab_encode_chars(self):
        sentence = ' '.join(['th', 'thhhhh', chr(256) + 't'])
        char_ids = self.vocab.encode_chars(sentence)
        expected = np.array(
            [[258, 256, 259, 260, 260],
            [258, 116, 104, 259, 260],
            [258, 116, 104, 104, 259],
            [258, 196, 128, 116, 259],
            [258, 257, 259, 260, 260]], dtype=np.int32)
        self.assertTrue((char_ids == expected).all())

    def test_vocab_encode_chars_reverse(self):
        sentence = ' '.join(reversed(['th', 'thhhhh', chr(256) + 't']))
        vocab = UnicodeCharsVocabulary(self._tmp, 5)
        char_ids = vocab.encode_chars(sentence, reverse=True)
        expected = np.array(
            [[258, 256, 259, 260, 260],
            [258, 116, 104, 259, 260],
            [258, 116, 104, 104, 259],
            [258, 196, 128, 116, 259],
            [258, 257, 259, 260, 260]], dtype=np.int32)[::-1, :]
        self.assertTrue((char_ids == expected).all())

    def tearDown(self):
        os.remove(self._tmp)


class TestBatcher(unittest.TestCase):
    def setUp(self):
        self._expected_char_ids = np.array(
                    [
                        [
                            [
                                259, 257, 260, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261
                            ], [
                                259, 85, 105, 102, 260, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261
                            ], [
                                259, 103, 106, 115, 116, 117, 260, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261
                            ], [
                                259, 116, 102, 111, 117, 102, 111, 100, 102,
                                260, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261
                            ], [
                                259, 258, 260, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261
                            ]
                        ], [
                            [
                                259, 257, 260, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261
                            ], [
                                259, 84, 102, 100, 112, 111, 101, 260, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261
                            ], [
                                259, 47, 260, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261
                            ], [
                                259, 258, 260, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261, 261, 261, 261, 261,
                                261, 261, 261, 261, 261
                            ], [
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                0, 0
                            ]
                        ]
                    ]
                )

    def test_batch_sentences(self):
        batcher = Batcher(os.path.join(DATA_FIXTURES, 'vocab_test.txt'), 50)
        sentences = [['The', 'first', 'sentence'], ['Second', '.']]
        x_char_ids = batcher.batch_sentences(sentences)

        self.assertTrue((x_char_ids == self._expected_char_ids).all())


class TestTokenBatcher(unittest.TestCase):
    def test_token_batcher(self):
        batcher = TokenBatcher(os.path.join(DATA_FIXTURES, 'vocab_test.txt'))
        sentences = [['The', 'first', '.'], ['It', 'said']]
        x_token_ids = batcher.batch_sentences(sentences)

        expected_ids = np.array([
            [2, 18, 75, 6, 1],
            [2, 67, 21, 1, 0]
        ])
        self.assertTrue((x_token_ids == expected_ids).all())


class TestLMDataset(unittest.TestCase):
    def setUp(self):
        sentences = ['the unknown .', 'th .', 'the']
        (_, tmp_train) = tempfile.mkstemp()
        with open(tmp_train, 'w') as fout:
            fout.write('\n'.join(sentences))

        words = ['<S>', '</S>', '<UNK>', 'the', '.', chr(256) + 't']
        (_, tmp_vocab) = tempfile.mkstemp()
        with open(tmp_vocab, 'w') as fout:
            fout.write('\n'.join(words))

        self._tmp_train = tmp_train
        self._tmp_vocab = tmp_vocab

    def _expected(self, reverse, chars, bidirectional=False):
        ret_forward = [
            {'tokens_characters':
                np.array(
                    [[[258, 256, 259, 260, 260],
                    [258, 116, 104, 101, 259],
                    [258, 117, 110, 107, 259]],
                
                    [[258, 256, 259, 260, 260],
                    [258, 116, 104, 259, 260],
                    [258,  46, 259, 260, 260]]], dtype=np.int32),
            'token_ids':
                np.array(
                    [[0, 3, 2],
                    [0, 2, 4]], dtype=np.int32),
             'next_token_id':
                 np.array(
                    [[3, 2, 4],
                    [2, 4, 1]], dtype=np.int32)},

            {'tokens_characters':
                np.array(
                    [[[258,  46, 259, 260, 260],
                    [258, 256, 259, 260, 260],
                    [258, 116, 104, 101, 259]],

                    [[258, 256, 259, 260, 260],
                    [258, 116, 104, 101, 259],
                    [258, 117, 110, 107, 259]]], dtype=np.int32),
             'token_ids':
                 np.array(
                    [[4, 0, 3],
                    [0, 3, 2]], dtype=np.int32),
             'next_token_id': 
                np.array(
                    [[1, 3, 1],
                    [3, 2, 4]], dtype=np.int32)}]

        ret_reverse = [
             {'tokens_characters': np.array([[[258, 257, 259, 260, 260],
                    [258,  46, 259, 260, 260],
                    [258, 117, 110, 107, 259]],
            
                   [[258, 257, 259, 260, 260],
                    [258,  46, 259, 260, 260],
                    [258, 116, 104, 259, 260]]], dtype=np.int32),
            'next_token_id': np.array([[4, 2, 3],
                   [4, 2, 0]], dtype=np.int32),
            'token_ids': np.array([[1, 4, 2],
                   [1, 4, 2]], dtype=np.int32)},
            
            {'tokens_characters': np.array([[[258, 116, 104, 101, 259],
                    [258, 257, 259, 260, 260],
                    [258, 116, 104, 101, 259]],
            
                   [[258, 257, 259, 260, 260],
                    [258,  46, 259, 260, 260],
                    [258, 117, 110, 107, 259]]], dtype=np.int32),
            'next_token_id': np.array([[0, 3, 0],
                   [4, 2, 3]], dtype=np.int32),
            'token_ids': np.array([[3, 1, 3],
                   [1, 4, 2]], dtype=np.int32)}]

        if bidirectional:
            expected = []
            for f, r in zip(ret_forward, ret_reverse):
                batch = dict(f)
                for k, v in r.items():
                    batch[k + '_reverse'] = v
                expected.append(batch)
        elif reverse:
            expected = ret_reverse
        else:
            expected = ret_forward

        if not chars:
            # set 'tokens_characters' key to None
            ret = []
            for e in expected:
                e['tokens_characters'] = None
                if 'tokens_characters_reverse' in e:
                    e['tokens_characters_reverse'] = None
                ret.append(e)
        else:
            ret = expected

        return ret

    def _load_data(self, reverse, chars, bidirectional=False):
        if chars:
            vocab = UnicodeCharsVocabulary(self._tmp_vocab, 5)
        else:
            vocab = Vocabulary(self._tmp_vocab)

        if not bidirectional:
            data = LMDataset(self._tmp_train, vocab, reverse=reverse)
        else:
            data = BidirectionalLMDataset(self._tmp_train, vocab)

        return data

    def _compare(self, actual, expected):
        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(sorted(list(a.keys())), sorted(list(e.keys())))
            for k in a.keys():
                if a[k] is not None:
                    self.assertTrue(np.all(a[k] == e[k]))
                else:
                    self.assertEqual(a[k], e[k])

    def _get_batches(self, *args, **kwargs):
        data = self._load_data(*args, **kwargs)
        batches = []
        np.random.seed(5)
        for i, batch in enumerate(data.iter_batches(2, 3)):
            batches.append(batch)
            if i == 1:
                break
        return batches

    def test_lm_dataset(self):
        batches = self._get_batches(False, True)
        expected = self._expected(False, True)
        self._compare(expected, batches)

    def test_lm_dataset_reverse(self):
        batches = self._get_batches(True, True)
        expected = self._expected(True, True)
        self._compare(expected, batches)

    def test_lm_dataset_no_char(self):
        batches = self._get_batches(False, False)
        expected = self._expected(False, False)
        self._compare(expected, batches)

    def test_lm_dataset_no_char_reverse(self):
        batches = self._get_batches(True, False)
        expected = self._expected(True, False)
        self._compare(expected, batches)

    def test_bi_lm_dataset(self):
        for a1 in [True, False]:
            for a2 in [True, False]:
                batches = self._get_batches(a1, a2, True)
                expected = self._expected(a1, a2, True)
            self._compare(expected, batches)


if __name__ == '__main__':
    unittest.main()



