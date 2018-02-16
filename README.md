# bilm-tf
Tensorflow implementation of the pretrained biLM used to compute ELMo
representations from ["Deep contextualized word representations"](http://arxiv.org/abs/1802.05365).

We also have a pytorch implementation available in [AllenNLP](http://allennlp.org/).

## Installing
Install python version 3.5 or later, tensorflow version 1.2 and h5py:

```
pip install tensorflow-gpu==1.2 h5py
python setup.py install
```

Ensure the tests pass in your environment by running:
```
python -m unittest discover tests/
```

Download the pretrained model consisting of an options file and the weight file:

* [options file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json)
* [weight file](https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5)

## Installing with Docker

To run the image, you must use nvidia-docker, because this repository
requires GPUs.
```
sudo nvidia-docker run -t allenai/bilm-tf:acc2f81d97fb2edd5174c797ae8ca415501deab1-GPU
```

## Usage overview
There are three ways to integrate ELMo representations into a downstream task, depending on your use case.

1. Compute representations on the fly from raw text using character input.  This is the most general method and will handle any input text.  It is also the most computationally expensive.
2. Precompute and cache the context independent token representations, then compute context dependent representations using the biLSTMs for input data.  This method is less computationally expensive then #1, but is only applicable with a fixed, prescribed vocabulary.
3.  Precompute the representations for your entire dataset and save to a file.

We have used all of these methods in the past for various use cases.  #1 is necessary for evaluating at test time on unseen data (e.g. public SQuAD leaderboard). #2 is a good compromise for large datasets where the size of the file in #3 is unfeasible (SNLI, SQuAD).  #3 is a good choice for smaller datasets or in cases where you'd like to use ELMo in other frameworks.

In all cases, the process roughly follows the same steps.
First, create a `Batcher` (or `TokenBatcher` for #2) to translate tokenized strings to numpy arrays of character (or token) ids.
Then, load the pretrained ELMo model (class `BidirectionalLanguageModel`).
Finally, for steps #1 and #2 use `weight_layers` to compute the final ELMo representations.
For #3, use `BidirectionalLanguageModel` to write all the intermediate layers to a file.

#### Shape conventions
Each tokenized sentence is a list of `str`, with a batch of sentences
a list of tokenized sentences (`List[List[str]]`).

The `Batcher` packs these into a shape
`(n_sentences, max_sentence_length + 2, 50)` numpy array of character
ids, padding on the right with 0 ids for sentences less then the maximum
length.  The first and last tokens for each sentence are special
begin and end of sentence ids added by the `Batcher`.

The input character id placeholder can be dimensioned `(None, None, 50)`,
with both the batch dimension (axis=0) and time dimension (axis=1) determined
for each batch, up the the maximum batch size specified in the
`BidirectionalLanguageModel` constructor.

After running inference with the batch, the return biLM embeddings are
a numpy array with shape `(n_sentences, 3, max_sentence_length, 1024)`,
after removing the special begin/end tokens.

#### Vocabulary file
The `Batcher` takes a vocabulary file as input for efficency.  This is a
text file, with one token per line, separated by newlines (`\n`).
Each token in the vocabulary is cached as the appropriate 50 character id
sequence once.  Since the model is completely character based, tokens not in
the vocabulary file are handled appropriately at run time, with a slight
decrease in run time.  It is recommended to always include the special
`<S>` and `</S>` tokens (case sensitive) in the vocabulary file.

### ELMo with character input

See `usage_character.py` for a detailed usage example.

### ELMo with pre-computed and cached context independent token representations
To speed up model inference with a fixed, specified vocabulary, it is
possible to pre-compute the context independent token representations,
write them to a file, and re-use them for inference.  Note that we don't
support falling back to character inputs for out-of-vocabulary words,
so this should only be used when the biLM is used to compute embeddings
for input with a fixed, defined vocabulary.

To use this option:

1.  First create a vocabulary file with all of the unique tokens in your
dataset and add the special `<S>` and `</S>` tokens.
2.  Run `dump_token_embeddings` with the full model to write the token
embeddings to a hdf5 file.
3.  Use `TokenBatcher` (instead of `Batcher`) with your vocabulary file,
and pass `use_token_inputs=False` and the name of the output file from step
2 to the `BidirectonalLanguageModel` constructor.

See `usage_token.py` for a detailed usage example.

### Dumping biLM embeddings for an entire dataset to a single file.

To take this option, create a text file with your tokenized dataset.  Each line is one tokenized sentence (whitespace separated).  Then use `dump_bilm_embeddings`.

The output file is `hdf5` format.  Each sentence in the input data is stored as a dataset with key `str(sentence_id)` where `sentence_id` is the line number in the dataset file (indexed from 0).
The embeddings for each sentence are a shape (3, n_tokens, 1024) array.

See `usage_cached.py` for a detailed example.

