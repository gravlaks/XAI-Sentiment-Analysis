"""
File name: emb.py

Creation Date: Mi 17 Feb 2021

Description: Embedding layer

"""

# Instructions
# -----------------------------------------------------------------------------
# Run emb.py with -m, -g and -p for model choice, glover file and processed file. Glover file used is Example file used from:https://www.kaggle.com/watts2/glove6b50dtxt  
# The second file is the preprocessed file we get from pre.py.
# Model choices are wither 0 for keras or 1 for xgboost. 

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

# Keras model test
# -----------------------------------------------------------------------------
from time import time
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

strategy = tf.distribute.get_strategy()

# xgboost model test
# -----------------------------------------------------------------------------
from xgboost import XGBClassifier
import xgboost as xgb 

# Local Application Modules
# -----------------------------------------------------------------------------
from parse import load_data

#var definition
MAX_WORDS = 5000
MAXLEN = 100


def get_embeddings_index(dat):
    embeddings_index = {}
    with open(dat, 'r',encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs
    
    print("Glove data loaded")
    return embeddings_index

def get_tokenizer(prepoc):
    df = load_data(prepoc)
    tweets = df['tweet']

    #Tokenizer magic
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(tweets)
    #temporary to test out the model 
    tokenized_tweets = tokenizer.texts_to_sequences(tweets)
    #remove this return sequence. for testing purposes. 
    return tokenizer,tokenized_tweets,df

##bread and butter of this whole operation. 
def get_embedding_matrix(glove, prepoc):
    #call the tokenizer
    word_idx = get_tokenizer(prepoc)[0].word_index 
    #Embeding bart :)
    embeddings_index = get_embeddings_index(glove)

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
    
    return  num_words, EMBEDDING_DIM, embedding_matrix

#Keras embedding layer.
def get_keras_embeddings_layer(glove, prepoc):
    num_words,EMBEDDING_DIM,embedding_matrix = get_embedding_matrix(glove, prepoc)
    MAX_SEQUENCE_LENGTH = 100
    
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    return  embedding_layer

##cgboost verison. sentence_feature_v2 returns the mean of the embedding matrix. This is the only thing necessary to create the training and test data. 
def sentence_features_v2(s, embedding_matrix,emb_size):
    M=np.array([embedding_matrix[w] for w in s])
    return M.mean(axis=0)

#test out if the embeding matrix works with xgboost. Simple model. 
def xgboost_model_test(glove, prepoc):
    
    split = 4000
    padding_type='post'
    trunc_type = 'post'
    tokenized_tweets = get_tokenizer(prepoc)[1]
    padded = pad_sequences(tokenized_tweets, maxlen=MAXLEN, padding=padding_type, truncating=trunc_type)
    training_tweets = padded[:split]
    test_tweets = padded[split:]
    test_labels = get_tokenizer(prepoc)[2]['target'][split:]
    training_labels = get_tokenizer(prepoc)[2]['target'][:split]
    
    num_words,EMBEDDING_DIM, embedding_matrix=get_embedding_matrix(glove, prepoc)

    x_train = np.array([sentence_features_v2(x,embedding_matrix,EMBEDDING_DIM) for x in training_tweets])
    x_test = np.array([sentence_features_v2(x,embedding_matrix,EMBEDDING_DIM) for x in test_tweets])

    # fit model no training data
    model = XGBClassifier()
    xgb_pars = {"min_child_weight": 50, "eta": 0.05, "max_depth": 8,
            "subsample": 0.8, "silent" : 1, "nthread": 4,
            "eval_metric": "mlogloss", "objective": "multi:softmax", "num_class": 2} #prøve å fikse det med for mange labels

    d_train = xgb.DMatrix(x_train, label=training_labels > 0)
    d_val = xgb.DMatrix(x_test, label=test_labels > 0)
    watchlist = [(d_train, 'train'), (d_val, 'valid')]

    bst = xgb.train(xgb_pars, d_train, 400, watchlist, early_stopping_rounds=50, verbose_eval=50)
    
    print("Training Complete")


#Test out if keras is working, simple model. 
def keras_model_test(glove, prepoc):

    split = 4000
    padding_type='post'
    trunc_type = 'post'
    tokenized_tweets = get_tokenizer(prepoc)[1]
    padded = pad_sequences(tokenized_tweets, maxlen=MAXLEN, padding=padding_type, truncating=trunc_type)
    training_tweets = padded[:split]
    test_tweets = padded[split:]
    test_labels = get_tokenizer(prepoc)[2]['target'][split:]
    training_labels = get_tokenizer(prepoc)[2]['target'][:split]

    with strategy.scope():    
    
        model = tf.keras.Sequential([
            get_keras_embeddings_layer(glove, prepoc),
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', default=0,
                        help='Choose model type to use, 0 for keras and 1 for xgboost. Default is Keras')
    parser.add_argument('-g', '--glove', default='./data/glove.6B.50d.txt',
                        help='The glover twitter file to train glover ')
    parser.add_argument('-p', '--prepoc',
                        help='The preprocessed file from running pre.py. ')

    args = parser.parse_args()

    if not args.glove or not args.prepoc:
        print('We need at least the preprocessed file boys. Try again with -p ***')
        exit()
    elif(int(args.model) == 1):
        xgboost_model_test(args.glove, args.prepoc)
    else:
        keras_model_test(args.glove, args.prepoc)

