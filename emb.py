"""
File name: emb.py

Creation Date: Mi 17 Feb 2021

Description: Embedding layer

"""

# Instructions
# -----------------------------------------------------------------------------
# Run get_keras_embeddings_layer to get Keras Embedding Layer

# Python Libraries
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.initializers import Constant
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Local Application Modules
# -----------------------------------------------------------------------------
# TODO: cleanup
# from parse import load_data


def get_embeddings_index(dat):
    embeddings_index = {}
    with open(dat, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs

    print("Glove data loaded")
    return embeddings_index

# TODO: cleanup
# def get_tweets(prepoc):
#     df = load_data(prepoc)
#     # df = pd.read_csv(prepoc, converters={'tweet': eval})
#     return df['tweet']


def get_embedding_matrix(glove, tweets, tokenizer):
    """
    Input parameters: pretrained glove file, preprocessed tweets
    and an unfitted tokenizer. 


    Returns a dictionary where the indexes are the tokens of each word and
    their value is the corresponding word vector's glove embedding vector.
    """
    # TODO: cleanup
    # tweets = get_tweets(prepoc)
    # print("got tweets")

    embeddings_index = get_embeddings_index(glove)
    EMBEDDING_DIM = embeddings_index.get('a').shape[0]

    #tokenizer.num_words = MAX_WORDS
    tokenizer.fit_on_texts(tweets)
    word_idx = tokenizer.word_index

    num_words = len(word_idx)+1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    not_found_count = 0
    for word, i in word_idx.items():
        # This references the loaded embeddings dictionary
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            not_found_count += 1
    print(f"Words not found {not_found_count}")

    return embedding_matrix

# Keras embedding layer.


def get_keras_embedding_layer(glove_file, tweets, tokenizer):
    emb_matrix = get_embedding_matrix(glove_file, tweets, tokenizer)

    # TODO this is probably overkill, find average and max length in our vocab then set something shorter
    MAX_SEQUENCE_LENGTH = 100

    emb_dim = len(emb_matrix[1])
    num_words = len(emb_matrix)
    embedding_layer = Embedding(num_words,
                                emb_dim,
                                embeddings_initializer=Constant(emb_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    return embedding_layer


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--glove', default='./data/glove.6B.50d.txt',
                        help='The glover twitter file to train glover ')
    parser.add_argument('-p', '--prepoc',
                        help='The preprocessed file from running pre.py. ')
    args = parser.parse_args()

    if not args.glove or not args.prepoc:
        print('We need at least the preprocessed file boys. Try again with -p ***')
        exit()
    else:

        tokenizer = Tokenizer(num_words=None,
                              lower=True, split=' ', oov_token="UNK")
        emb_layer = get_keras_embeddings_layer(
            args.glove, args.prepoc, tokenizer)
