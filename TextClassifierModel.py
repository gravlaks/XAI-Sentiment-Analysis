#!/bin/bash
"""
File name: model.py

Creation Date: Wed 10 Mar 2021

Description:

"""

# Python Libraries
# -----------------------------------------------------------------------------------------

import json
import os
import pickle
from shutil import rmtree

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from emb import get_keras_embedding_layer

# Local Application Modules
# -----------------------------------------------------------------------------------------


class KerasTextClassifier():
    '''Wrapper class for keras text classification models that takes raw text as input.'''

    def __init__(self, max_words=30000, input_length=100):
        EMB_MAX_WORDS = 5000

        self.tokenizer = Tokenizer(num_words=EMB_MAX_WORDS, lower=False, split=' ', oov_token="UNK")

        self.input_length = input_length

    def get_model(self, emb_layer):
        n_classes = 2

        sequential_model = tf.keras.Sequential([
            emb_layer,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(128, activation='relu',
                                  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            tf.keras.layers.Dense(256, activation='sigmoid'),
            tf.keras.layers.Dense(n_classes, activation='softmax')
        ])

        inp = tf.keras.Input(shape=(None,), name="title"
                             )
        embedded = emb_layer(inp)
        lstm = tf.keras.layers.LSTM(128)(embedded)
        flatten = tf.keras.layers.Flatten()(lstm)
        dense = tf.keras.layers.Dense(
            n_classes, activation='sigmoid')(flatten)
        lstm_model = tf.keras.Model(
            inputs=inp,
            outputs=dense
        )

        model = sequential_model

        model.compile(optimizer="adam",
                      loss="binary_crossentropy",
                      metrics=["accuracy"])
        print(model.summary())
        return model

    def _get_sequences(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.input_length, value=0)

    def fit(self, X_train, y_train, validation_data, epochs, batch_size, verbose):
        '''
        Fit the vocabulary and the model.

        :params:
        X: list of list of words.
        y: labels.
        '''

        seqs = self._get_sequences(X_train)
        seqs_val = self._get_sequences(validation_data[0])

        # print(seqs[0])
        # print(type(seqs), type(seqs[0]))

        # seqs = np.array([list(x) for x in seqs])

        return self.model.fit(seqs, y_train, validation_data=(seqs_val, validation_data[1]), batch_size=batch_size, epochs=epochs, verbose=verbose)

    def predict_proba(self, X, y=None):

        seqs = self._get_sequences(X)

        return self.model.predict(seqs)

    def predict(self, X, y=None):
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, directory, overwrite=False):
        save_model_keras(self, directory, overwrite)


def build_model_keras(tokenizer, emb_layer):
    print("building model")
    model = KerasTextClassifier(tokenizer, emb_layer)

    return model


def save_model_keras(model, directory, overwrite=False):
    if os.path.exists(directory):
        if overwrite:
            print('First, removing existing model')
            rmtree(directory)
        else:
            raise TypeError('Directory exists!')

    print('Saving model to', directory)
    os.makedirs(directory, exist_ok=True)
    config_path, tokenizer_path, model_path = get_model_paths(directory)

    print('Creating config file', config_path)
    with open(config_path, 'w') as out_file:
        json.dump({'input_length': model.input_length}, out_file)

    print('Creating tokenizer file', tokenizer_path)
    with open(tokenizer_path, 'wb') as out_file:
        pickle.dump(model.tokenizer, out_file, pickle.HIGHEST_PROTOCOL)

    print('Creating keras model file', model_path)
    model.model.save(model_path)


def load_classifier(model_path=None, glove_file = None, data=None):

    text_classifier_path = 'models/text_classifier.pkl'
    if not os.path.isfile(text_classifier_path):
        text_classifier = KerasTextClassifier()

        #Save empty text classifier
        #with open(text_classifier_path, 'wb') as out_file:
            #pickle.dump(text_classifier, out_file, pickle.HIGHEST_PROTOCOL)

    else:
        with open(text_classifier_path, 'rb') as in_file:
            text_classifier = pickle.load(in_file)


    if model_path is None:
        tokenizer = text_classifier.tokenizer
        emb_layer = get_keras_embedding_layer(glove_file, data['tweet'], tokenizer)
        text_classifier.model = text_classifier.get_model(emb_layer)
        model_path = 'models/untrained'
        #tf.keras.models.save_model(text_classifier.model, model_path)
    else: 
        text_classifier.model = tf.keras.models.load_model(model_path)
        text_classifier.model.layers[0].trainable=False
    
    save_classifier(text_classifier, model_path)
    return text_classifier

def save_classifier(text_classifier, model_path):
    
    tf.keras.models.save_model(text_classifier.model, model_path)
    
    text_classifier.model = None

    text_classifier_path = 'models/text_classifier.pkl'
    with open(text_classifier_path, 'wb') as out_file:
        pickle.dump(text_classifier, out_file, pickle.HIGHEST_PROTOCOL)

    text_classifier.model = tf.keras.models.load_model(model_path)


