'''
ELMo usage example with pre-computed and cached context independent
token representations

Below, we show usage for SQuAD where each input example consists of both
a question and a paragraph of context.
'''

import tensorflow as tf
import os
from bilm import TokenBatcher, BidirectionalLanguageModel, weight_layers, \
    dump_token_embeddings

# Our small dataset.
raw_context = [
    'Pretrained biLMs compute representations useful for NLP tasks .',
    'They give state of the art performance for many tasks .'
]
tokenized_context = [sentence.split() for sentence in raw_context]
tokenized_question = [
    ['What', 'are', 'biLMs', 'useful', 'for', '?'],
]

# Create the vocabulary file with all unique tokens and
# the special <S>, </S> tokens (case sensitive).
all_tokens = set(['<S>', '</S>'] + tokenized_question[0])
for context_sentence in tokenized_context:
    for token in context_sentence:
        all_tokens.add(token)
vocab_file = 'vocab_small.txt'
with open(vocab_file, 'w') as fout:
    fout.write('\n'.join(all_tokens))

# Location of pretrained LM.  Here we use the test fixtures.
datadir = os.path.join('tests', 'fixtures', 'model')
options_file = os.path.join(datadir, 'options.json')
weight_file = os.path.join(datadir, 'lm_weights.hdf5')

# Dump the token embeddings to a file. Run this once for your dataset.
token_embedding_file = 'elmo_token_embeddings.hdf5'
dump_token_embeddings(
    vocab_file, options_file, weight_file, token_embedding_file
)
tf.reset_default_graph()



## Now we can do inference.
# Create a TokenBatcher to map text to token ids.
batcher = TokenBatcher(vocab_file)

# Input placeholders to the biLM.
context_token_ids = tf.placeholder('int32', shape=(None, None))
question_token_ids = tf.placeholder('int32', shape=(None, None))

# Build the biLM graph.
bilm = BidirectionalLanguageModel(
    options_file,
    weight_file,
    use_character_inputs=False,
    embedding_weight_file=token_embedding_file
)

# Get ops to compute the LM embeddings.
context_embeddings_op = bilm(context_token_ids)
question_embeddings_op = bilm(question_token_ids)

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


with tf.Session() as sess:
    # It is necessary to initialize variables once before running inference.
    sess.run(tf.global_variables_initializer())

    # Create batches of data.
    context_ids = batcher.batch_sentences(tokenized_context)
    question_ids = batcher.batch_sentences(tokenized_question)

    # Compute ELMo representations (here for the input only, for simplicity).
    elmo_context_input_, elmo_question_input_ = sess.run(
        [elmo_context_input['weighted_op'], elmo_question_input['weighted_op']],
        feed_dict={context_token_ids: context_ids,
                   question_token_ids: question_ids}
    )

