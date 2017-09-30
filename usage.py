
import tensorflow as tf
import os
from bilm import Batcher, BidirectionalLanguageModel

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

# Build the graphs.  It is necessary to specify a reuse
# variable scope (with name='') for every placeholder except the first.
# All variables names for the bidirectional LMs are prefixed with 'bilm/'.
context_model = BidirectionalLanguageModel(
    options_file, weight_file, context_character_ids)
with tf.variable_scope('', reuse=True):
    question_model = BidirectionalLanguageModel(
        options_file, weight_file, question_character_ids)

# Get ops to compute the LM embeddings.
context_embeddings_op = context_model.get_ops()['lm_embeddings']
question_embeddings_op = question_model.get_ops()['lm_embeddings']

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

    # Compute LM embeddings.
    context_embeddings, question_embeddings = sess.run(
        [context_embeddings_op, question_embeddings_op],
        feed_dict={context_character_ids: context_ids,
                   question_character_ids: question_ids}
    )

