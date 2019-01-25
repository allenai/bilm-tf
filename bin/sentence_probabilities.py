"""
Compute the probabilities for sentences by the language model used to train ELMO embeddings.

This is very similar to run_test and should probably be refactored together with it.
"""

import argparse
import logging

from bilm.training import test, load_options_latest_checkpoint, load_vocab, sentence_probabilities
from bilm.data import LMDataset, BidirectionalLMDataset


def main(args):
    options, ckpt_file = load_options_latest_checkpoint(args.save_dir)

    logging.getLogger().setLevel(logging.INFO)

    # load the vocab
    if 'char_cnn' in options:
        max_word_length = options['char_cnn']['max_characters_per_token']
    else:
        max_word_length = None
    # vocab = load_vocab(args.vocab_file, max_word_length)

    sentence_file = args.test_prefix

    kwargs = {
        'test': True,
        'shuffle_on_load': False,
    }

    # if options.get('bidirectional'):
    #     data = BidirectionalLMDataset(test_prefix, vocab, **kwargs)
    # else:
    #     data = LMDataset(test_prefix, vocab, **kwargs)

#    sentence_probabilities(options, ckpt_file, data, batch_size=args.batch_size)
    if args.burn_in_text:
        with open(args.burn_in_text) as burn_in_text_inp:
            burn_in_text = list(burn_in_text_inp)
        print("Got burn in text")
    else:
        burn_in_text = None

    sentence_probabilities(options, ckpt_file, sentence_file,
                           args.vocab_file, batch_size=args.batch_size,
                           burn_in_text=burn_in_text)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute sentence probabilities')
    parser.add_argument('--save_dir', help='Location of checkpoint files')
    parser.add_argument('--vocab_file', help='Vocabulary file')
    parser.add_argument('--test_prefix', help='Prefix for test files')
    parser.add_argument('--burn_in_text', help="File of burn in text to determine initial LSTM "
                                               "states", default=None)
    parser.add_argument('--batch_size',
        type=int, default=256,
        help='Batch size')

    args = parser.parse_args()
    main(args)

