
import unittest
import tempfile
import os
import numpy as np

from bilm.data import UnicodeCharsVocabulary, Vocabulary, Batcher, TokenBatcher

FIXTURES = 'tests/fixtures/data/'


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
        batcher = Batcher(os.path.join(FIXTURES, 'vocab_test.txt'), 50)
        sentences = [['The', 'first', 'sentence'], ['Second', '.']]
        x_char_ids = batcher.batch_sentences(sentences)

        self.assertTrue((x_char_ids == self._expected_char_ids).all())


class TestTokenBatcher(unittest.TestCase):
    def test_token_batcher(self):
        batcher = TokenBatcher(os.path.join(FIXTURES, 'vocab_test.txt'))
        sentences = [['The', 'first', '.'], ['It', 'said']]
        x_token_ids = batcher.batch_sentences(sentences)

        expected_ids = np.array([
            [2, 18, 75, 6, 1],
            [2, 67, 21, 1, 0]
        ])
        self.assertTrue((x_token_ids == expected_ids).all())


if __name__ == '__main__':
    unittest.main()


