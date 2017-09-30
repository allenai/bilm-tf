
import numpy as np
import tensorflow as tf
import h5py
import json
import re

DTYPE = 'float32'
DTYPE_INT = 'int64'


class BidirectionalLanguageModel(object):
    def __init__(
            self,
            options_file: str,
            weight_file: str,
            character_ids_placeholder,
            max_batch_size=128
        ):
        '''
        Creates the language model computational graph and loads weights

        options_file: location of the json formatted file with
                      LM hyperparameters
        weight_file: location of the hdf5 file with LM weights
        character_ids_placeholder: a tf.placeholder of type int32 and shape
            (None, None, max_characters_per_token) holding the input
            character ids for a batch
        max_batch_size: the maximum allowable batch size 
        '''
        with open(options_file, 'r') as fin:
            options = json.load(fin)

        self._lm_graph = BidirectionalLanguageModelGraph(options, weight_file,
            character_ids_placeholder, max_batch_size)
        self._max_token_length = options['char_cnn']['max_characters_per_token']
        self._ops = None

    def get_ops(self):
        '''
        returns a dictionary with tensorflow ops:
            {'lm_embeddings': embedding_op,
             'lengths': sequence_lengths_op}

        embedding_op computes the LM embeddings and is shape
            (None, 3, None, 1024)
        lengths_op computes the sequence lengths and is shape (None, )
        '''
        if self._ops is None:
            self._build_ops()
        return self._ops

    def _build_ops(self):
        with tf.control_dependencies([self._lm_graph.update_state_op]):
            # get the LM embeddings
            token_embeddings = self._lm_graph.embedding
            layers = [
                tf.concat([token_embeddings, token_embeddings], axis=2)
            ]

            n_lm_layers = len(self._lm_graph.lstm_outputs['forward'])
            for i in range(n_lm_layers):
                layers.append(
                    tf.concat(
                        [self._lm_graph.lstm_outputs['forward'][i],
                         self._lm_graph.lstm_outputs['backward'][i]],
                        axis=-1
                    )
                )

            # The layers include the BOS/EOS tokens.  Remove them
            sequence_length_wo_bos_eos = self._lm_graph.sequence_lengths - 2
            layers_without_bos_eos = []
            for layer in layers:
                layer_wo_bos_eos = layer[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos, 
                    self._lm_graph.sequence_lengths - 1,
                    seq_axis=1,
                    batch_axis=0,
                )
                layer_wo_bos_eos = layer_wo_bos_eos[:, 1:, :]
                layer_wo_bos_eos = tf.reverse_sequence(
                    layer_wo_bos_eos,
                    sequence_length_wo_bos_eos,
                    seq_axis=1,
                    batch_axis=0,
                )
                layers_without_bos_eos.append(layer_wo_bos_eos)

            # concatenate the layers
            lm_embeddings = tf.concat(
                [tf.expand_dims(t, axis=1) for t in layers_without_bos_eos],
                axis=1
            )

        self._ops = {
            'lm_embeddings': lm_embeddings, 
            'lengths': sequence_length_wo_bos_eos
        }


def _pretrained_initializer(varname, weight_file):
    '''
    We'll stub out all the initializers in the pretrained LM with
    a function that loads the weights from the file
    '''
    weight_name_map = {}
    for i in range(2):
        for j in range(8):  # if we decide to add more layers
            root = 'RNN_{}/RNN/MultiRNNCell/Cell{}'.format(i, j)
            weight_name_map[root + '/rnn/lstm_cell/kernel'] = \
                root + '/LSTMCell/W_0'
            weight_name_map[root + '/rnn/lstm_cell/bias'] = \
                root + '/LSTMCell/B'
            weight_name_map[root + '/rnn/lstm_cell/projection/kernel'] = \
                root + '/LSTMCell/W_P_0'

    # convert the graph name to that in the checkpoint
    varname_in_file = varname[5:]
    if varname_in_file.startswith('RNN'):
        varname_in_file = weight_name_map[varname_in_file]

    with h5py.File(weight_file, 'r') as fin:
        if varname_in_file == 'char_embed':
            # Have added a special 0 index for padding not present
            # in the original model.
            char_embed_weights = fin[varname_in_file][...]
            weights = np.zeros(
                (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]),
                dtype=DTYPE
            )
            weights[1:, :] = char_embed_weights
        else:
            weights = fin[varname_in_file][...]

    # Tensorflow initializers are callables that accept a shape parameter
    # and some optional kwargs
    def ret(shape, **kwargs):
        if list(shape) != list(weights.shape):
            raise ValueError(
                "Invalid shape initializing {0}, got {1}, expected {2}".format(
                    varname_in_file, shape, weights.shape)
            )
        return weights

    return ret


class BidirectionalLanguageModelGraph(object):
    '''
    Creates the computational graph and holds the ops necessary for runnint
    a bidirectional language model
    '''
    def __init__(self, options, weight_file, tokens_characters,
                 max_batch_size=128):

        self.options = options
        self._max_batch_size = max_batch_size
        self.tokens_characters = tokens_characters

        # this custom_getter will make all variables not trainable and
        # override the default initializer
        def custom_getter(getter, name, *args, **kwargs):
            kwargs['trainable'] = False
            kwargs['initializer'] = _pretrained_initializer(name, weight_file)
            return getter(name, *args, **kwargs)

        with tf.variable_scope('bilm', custom_getter=custom_getter):
            self._build()

    def _build(self):
        self._build_word_char_embeddings()
        self._build_lstms()

    def _build_word_char_embeddings(self):
        '''
        options contains key 'char_cnn': {

        'n_characters': 60,

        # includes the start / end characters
        'max_characters_per_token': 17,

        'filters': [
            [1, 32],
            [2, 32],
            [3, 64],
            [4, 128],
            [5, 256],
            [6, 512],
            [7, 512]
        ],
        'activation': 'tanh',

        # for the character embedding
        'embedding': {'dim': 16}

        # for highway layers
        # if omitted, then no highway layers
        'n_highway': 2,
        }
        '''
        projection_dim = self.options['lstm']['projection_dim']

        cnn_options = self.options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        max_chars = cnn_options['max_characters_per_token']
        char_embed_dim = cnn_options['embedding']['dim']
        n_chars = cnn_options['n_characters']
        if cnn_options['activation'] == 'tanh':
            activation = tf.nn.tanh
        elif cnn_options['activation'] == 'relu':
            activation = tf.nn.relu

        # the character embeddings
        with tf.device("/cpu:0"):
            self.embedding_weights = tf.get_variable(
                    "char_embed", [n_chars, char_embed_dim],
                    dtype=DTYPE,
                    initializer=tf.random_uniform_initializer(-1.0, 1.0)
            )
            # shape (batch_size, unroll_steps, max_chars, embed_dim)
            self.char_embedding = tf.nn.embedding_lookup(self.embedding_weights,
                                                    self.tokens_characters)

        # the convolutions
        def make_convolutions(inp):
            with tf.variable_scope('CNN') as scope:
                convolutions = []
                for i, (width, num) in enumerate(filters):
                    if cnn_options['activation'] == 'relu':
                        # He initialization for ReLU activation
                        # with char embeddings init between -1 and 1
                        #w_init = tf.random_normal_initializer(
                        #    mean=0.0,
                        #    stddev=np.sqrt(2.0 / (width * char_embed_dim))
                        #)

                        # Kim et al 2015, +/- 0.05
                        w_init = tf.random_uniform_initializer(
                            minval=-0.05, maxval=0.05)
                    elif cnn_options['activation'] == 'tanh':
                        # glorot init
                        w_init = tf.random_normal_initializer(
                            mean=0.0,
                            stddev=np.sqrt(1.0 / (width * char_embed_dim))
                        )
                    w = tf.get_variable(
                        "W_cnn_%s" % i,
                        [1, width, char_embed_dim, num],
                        initializer=w_init,
                        dtype=DTYPE)
                    b = tf.get_variable(
                        "b_cnn_%s" % i, [num], dtype=DTYPE,
                        initializer=tf.constant_initializer(0.0))

                    conv = tf.nn.conv2d(
                            inp, w,
                            strides=[1, 1, 1, 1],
                            padding="VALID") + b
                    # now max pool
                    conv = tf.nn.max_pool(
                            conv, [1, 1, max_chars-width+1, 1],
                            [1, 1, 1, 1], 'VALID')

                    # activation
                    conv = activation(conv)
                    conv = tf.squeeze(conv, squeeze_dims=[2])

                    convolutions.append(conv)

            return tf.concat(convolutions, 2)

        embedding = make_convolutions(self.char_embedding)

        # for highway and projection layers
        n_highway = cnn_options.get('n_highway')
        use_highway = n_highway is not None and n_highway > 0
        use_proj = n_filters != projection_dim

        if use_highway or use_proj:
            #   reshape from (batch_size, n_tokens, dim) to (-1, dim)
            batch_size_n_tokens = tf.shape(embedding)[0:2]
            embedding = tf.reshape(embedding, [-1, n_filters])

        # set up weights for projection
        if use_proj:
            assert n_filters > projection_dim
            with tf.variable_scope('CNN_proj') as scope:
                    W_proj_cnn = tf.get_variable(
                        "W_proj", [n_filters, projection_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / n_filters)),
                        dtype=DTYPE)
                    b_proj_cnn = tf.get_variable(
                        "b_proj", [projection_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

        # apply highways layers
        def high(x, ww_carry, bb_carry, ww_tr, bb_tr):
            carry_gate = tf.nn.sigmoid(tf.matmul(x, ww_carry) + bb_carry)
            transform_gate = tf.nn.relu(tf.matmul(x, ww_tr) + bb_tr)
            return carry_gate * transform_gate + (1.0 - carry_gate) * x

        if use_highway:
            highway_dim = n_filters

            for i in range(n_highway):
                with tf.variable_scope('CNN_high_%s' % i) as scope:
                    W_carry = tf.get_variable(
                        'W_carry', [highway_dim, highway_dim],
                        # glorit init
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_carry = tf.get_variable(
                        'b_carry', [highway_dim],
                        initializer=tf.constant_initializer(-2.0),
                        dtype=DTYPE)
                    W_transform = tf.get_variable(
                        'W_transform', [highway_dim, highway_dim],
                        initializer=tf.random_normal_initializer(
                            mean=0.0, stddev=np.sqrt(1.0 / highway_dim)),
                        dtype=DTYPE)
                    b_transform = tf.get_variable(
                        'b_transform', [highway_dim],
                        initializer=tf.constant_initializer(0.0),
                        dtype=DTYPE)

                embedding = high(embedding, W_carry, b_carry,
                                 W_transform, b_transform)

        # finally project down if needed
        if use_proj:
            embedding = tf.matmul(embedding, W_proj_cnn) + b_proj_cnn

        # reshape back to (batch_size, tokens, dim)
        if use_highway or use_proj:
            shp = tf.concat([batch_size_n_tokens, [projection_dim]], axis=0)
            embedding = tf.reshape(embedding, shp)

        # at last assign attributes for remainder of the model
        self.embedding = embedding


    def _build_lstms(self):
        # now the LSTMs
        # these will collect the initial states for the forward
        #   (and reverse LSTMs if we are doing bidirectional)

        # parse the options
        lstm_dim = self.options['lstm']['dim']
        projection_dim = self.options['lstm']['projection_dim']
        n_lstm_layers = self.options['lstm'].get('n_layers', 1)
        cell_clip = self.options['lstm'].get('cell_clip')
        proj_clip = self.options['lstm'].get('proj_clip')
        use_skip_connections = self.options['lstm'].get(
                                            'use_skip_connections')
        if use_skip_connections:
            print("USING SKIP CONNECTIONS")

        #lstm_outputs = []
        #for lstm_num, lstm_input in enumerate(lstm_inputs):

        # the sequence lengths from input mask
        mask = tf.reduce_any(self.tokens_characters > 0, axis=2)
        sequence_lengths = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
        batch_size = tf.shape(sequence_lengths)[0]

        # for each direction, we'll store tensors for each layer
        self.lstm_outputs = {'forward': [], 'backward': []}
        self.lstm_state_sizes = {'forward': [], 'backward': []}
        self.lstm_init_states = {'forward': [], 'backward': []}
        self.lstm_final_states = {'forward': [], 'backward': []}

        update_ops = []
        for direction in ['forward', 'backward']:
            if direction == 'forward':
                layer_input = self.embedding
            else:
                layer_input = tf.reverse_sequence(
                    self.embedding,
                    sequence_lengths,
                    seq_axis=1,
                    batch_axis=0
                )

            for i in range(n_lstm_layers):
                if projection_dim < lstm_dim:
                    # are projecting down output
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                        lstm_dim, num_proj=projection_dim,
                        cell_clip=cell_clip, proj_clip=proj_clip)
                else:
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(
                            lstm_dim,
                            cell_clip=cell_clip, proj_clip=proj_clip)

                if use_skip_connections:
                    # ResidualWrapper adds inputs to outputs
                    if i == 0:
                        # don't add skip connection from token embedding to
                        # 1st layer output
                        pass
                    else:
                        # add a skip connection
                        lstm_cell = tf.nn.rnn_cell.ResidualWrapper(lstm_cell)

                # collect the input state, run the dynamic rnn, collect
                # the output
                state_size = lstm_cell.state_size
                # the LSTMs are stateful.  To support multiple batch sizes,
                # we'll allocate size for states up to max_batch_size,
                # then use the first batch_size entries for each batch
                init_states = [
                    tf.Variable(
                        tf.zeros([self._max_batch_size, dim]),
                        trainable=False
                    )
                    for dim in lstm_cell.state_size
                ]
                batch_init_states = [
                    state[:batch_size, :] for state in init_states
                ]

                if direction == 'forward':
                    i_direction = 0
                else:
                    i_direction = 1
                variable_scope_name = 'RNN_{0}/RNN/MultiRNNCell/Cell{1}'.format(
                    i_direction, i)
                with tf.variable_scope(variable_scope_name):
                    layer_output, final_state = tf.nn.dynamic_rnn(
                        lstm_cell,
                        layer_input,
                        sequence_length=sequence_lengths,
                        initial_state=tf.nn.rnn_cell.LSTMStateTuple(
                            *batch_init_states),
                    )

                self.lstm_state_sizes[direction].append(lstm_cell.state_size)
                self.lstm_init_states[direction].append(init_states)
                self.lstm_final_states[direction].append(final_state)
                if direction == 'forward':
                    self.lstm_outputs[direction].append(layer_output)
                else:
                    self.lstm_outputs[direction].append(
                        tf.reverse_sequence(
                            layer_output,
                            sequence_lengths,
                            seq_axis=1,
                            batch_axis=0
                        )
                    )

                with tf.control_dependencies([layer_output]):
                    # update the initial states
                    for i in range(2):
                        new_state = tf.concat(
                            [final_state[i][:batch_size, :],
                             init_states[i][batch_size:, :]], axis=0)
                        state_update_op = tf.assign(init_states[i], new_state)
                        update_ops.append(state_update_op)
    
                layer_input = layer_output

        self.mask = mask
        self.sequence_lengths = sequence_lengths
        self.update_state_op = tf.group(*update_ops)

