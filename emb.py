"""
File name: emb.py

Creation Date: Mi 17 Feb 2021

Description: Embedding layer

"""

# Python Libraries
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer

# Local Application Modules
# -----------------------------------------------------------------------------


def get_embeddings_index():
    embeddings_index = {}
    with open('data/glove.6B.50d.txt', 'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs
    
    print("Glove data loaded")
    return embeddings_index



def get_preprocessed_tweets(input_file):
    
    print("Fetching preprocessed tweets")

    df = pd.read_csv(input_file, header=None)
    return df


df = get_preprocessed_tweets('data/preproc_1000.csv')
labels = to_categorical(df.iloc[1:, 0], 3)
tweets = np.array(df.iloc[1:, 1])
print(tweets[0])

MAX_WORDS = 1000
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(tweets)
tweets = tokenizer.texts_to_sequences(tweets)
print(tweets[0])

word_idx = tokenizer.word_index
print(word['somehow'])


raise Exception


embeddings_index = get_embeddings_index()
EMBEDDING_DIM = embeddings_index.get('a').shape[0]


num_words = min(MAX_WORDS, len(word_idx))+1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_idx.items():
    if i > MAX_WORDS:
        continue
    embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print(embedding_matrix.shape)

print(embedding_matrix[0])
