'''
ELMo usage example with character inputs.

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import tensorflow as tf
import os
from bilm import Batcher, BidirectionalLanguageModel, weight_layers

# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('tests', 'fixtures', 'model')
vocab_file = os.path.join(datadir, 'vocab_test.txt')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')

# Create a Batcher to map text to character ids.
batcher = Batcher(vocab_file, 50)

# Input placeholders to the biLM.
context_character_ids = tf.placeholder('int32', shape=(None, None, 50))
question_character_ids = tf.placeholder('int32', shape=(None, None, 50))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(options_file, weight_file)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_character_ids)
question_embeddings_op = bilm(question_character_ids)

# Get an op to compute ELMo (weighted average of the internal biLM layers)
# Our SQuAD model includes ELMo at both the input and output layers
# of the task GRU, so we need 4x ELMo representations for the question
# and context at each of the input and output.
# We use the same ELMo weights for both the question and context
# at each of the input and output.
elmo_context_input = weight_layers('input', context_embeddings_op, l2_coef=0.0)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_input = weight_layers(
        'input', question_embeddings_op, l2_coef=0.0
    )

elmo_context_output = weight_layers(
    'output', context_embeddings_op, l2_coef=0.0
)
with tf.variable_scope('', reuse=True):
    # the reuse=True scope reuses weights from the context for the question
    elmo_question_output = weight_layers(
        'output', question_embeddings_op, l2_coef=0.0
    )


# Now we can compute embeddings.
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_question)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
        feed_dict={context_character_ids: context_ids,
                   question_character_ids: question_ids}
    )

