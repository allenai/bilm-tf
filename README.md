# bilm-tf
Tensorflow implementation of pretrained biLM from ["Deep contextualized word representations"](https://openreview.net/forum?id=S1p31z-Ab).

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

You can also run `bilm-tf` using Docker. From the cloned repository, run:
```
sudo docker build -t bilm-tf .
```
To run the image, you must use nvidia-docker, because this repository
requires GPUs.
```
sudo nvidia-docker run -t bilm-tf
```

TODO: Add public image once it is released.

## Usage overview
There are two main public classes: `Batcher` and `BidirectionalLanguageModel`.

The `Batcher` is used to convert lists of tokenized sentences to numpy
arrays of character ids for the model to run inference.

`BidirectionalLanguageModel` loads the pre-trained model, creates the
computational graph and returns ops that run inference.

See `usage.py` for a detailed usage example.

## Shape conventions
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

## Vocabulary file
The `Batcher` takes a vocabulary file as input for efficency.  This is a
text file, with one token per line, separated by newlines (`\n`).
Each token in the vocabulary is cached as the appropriate 50 character id
sequence once.  Since the model is completely character based, tokens not in
the vocabulary file are handled appropriately at run time, with a slight
decrease in run time.  It is recommended to always include the special
`<S>` and `</S>` tokens (case sensitive) in the vocabulary file.

## Optional mode with pre-computed token embeddings
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


