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
from keras.layers import Embedding
from keras.initializers import Constant
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Local Application Modules
# -----------------------------------------------------------------------------
from parse import load_data

strategy = tf.distribute.get_strategy()

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

    df = pd.read_csv(input_file, converters={'tweet': eval}) 
    #df["tweet"].apply(lambda tweet: tweet.split(' '))
    return df


df = load_data('data/prepoc1000.csv')
tweets = df['tweet']

#df['tweet'].apply(lambda tweet: list(tweet))
#training_labels = to_categorical(np.asarray(df['target']), 3)

print(tweets[1])
print(type(tweets[1]))
print(tweets[1][0])

MAX_WORDS = 5000
MAXLEN = 100
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(tweets)
tokenized_tweets = tokenizer.texts_to_sequences(tweets)
print(tweets[0])
print(tokenized_tweets[0])

word_idx = tokenizer.word_index

split = 500
padding_type='post'
trunc_type = 'post'
padded = pad_sequences(tokenized_tweets, maxlen=MAXLEN, padding=padding_type, truncating=trunc_type)
training_tweets = padded[:split]
training_labels = df['target'][:split]


embeddings_index = get_embeddings_index()







def cosine_similarity(u, v):

    distance = 0.0
    
    # the dot product between u and v 
    dot = np.dot(u, v)
    # the L2 norm of u 
    norm_u = np.sqrt(np.sum((u)**2))
    
    # the L2 norm of v 
    norm_v = np.sqrt(np.sum((v)**2))
    # the cosine similarity defined by formula 
    cosine_similarity = dot/(norm_u*norm_v)
    
    return cosine_similarity


EMBEDDING_DIM = embeddings_index.get('a').shape[0]


num_words = min(MAX_WORDS, len(word_idx))+1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
not_found_count = 0
for word, i in word_idx.items():
    if i > MAX_WORDS:
        continue
    embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary
    if embedding_vector is not None:
    
        
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
    else:
        not_found_count += 1
print(f"Words not found {not_found_count}")

print(embedding_matrix.shape)
MAX_SEQUENCE_LENGTH = 100
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)





from time import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

with strategy.scope():    
    
    model = tf.keras.Sequential([
        embedding_layer,
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Bidirectional(LSTM(units=64, return_sequences=True)),
        tf.keras.layers.Bidirectional(LSTM(units=128)),
        tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    
    num_epochs = 15
    
    training_padded = np.array(training_tweets)
    training_labels = np.array(training_labels)
    
    history = model.fit(training_padded, 
                        training_labels, 
                        epochs=num_epochs, 
                        validation_data=(training_padded, training_labels),
                        batch_size = 256,
                        verbose=1)
    
    print("Training Complete")
