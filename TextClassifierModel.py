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

# Local Application Modules
# -----------------------------------------------------------------------------------------


class KerasTextClassifier():
    '''Wrapper class for keras text classification models that takes raw text as input.'''

    def __init__(self, tokenizer, emb_layer, max_words=30000, input_length=100):
        if tokenizer is not None and emb_layer is not None:
            print("init")
            self.input_length = input_length
            self.model = self._get_model(emb_layer)
            self.tokenizer = tokenizer

    def _get_model(self, emb_layer):
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

    def fit(self, X, y, epochs, batch_size):
        '''
        Fit the vocabulary and the model.

        :params:
        X: list of texts.
        y: labels.
        '''

        seqs = self._get_sequences(X)

        return self.model.fit(seqs, y, batch_size=batch_size, epochs=epochs, validation_split=0.1)

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


def load_model_keras(directory):
    if not os.path.exists(directory):
        raise TypeError('Directory does not exist!')
    if not os.path.isdir(directory):
        raise TypeError('Path exists but is not a directory!')

    config_path, tokenizer_path, model_path = get_model_paths(directory)
    if not os.path.isfile(config_path):
        raise TypeError('No config file!')
    if not os.path.isfile(tokenizer_path):
        raise TypeError('No tokenizer file!')
    if not os.path.exists(model_path):
        raise TypeError('No keras model!')

    print('Loading model from', directory)
    model = KerasTextClassifier(None, None)

    print('Reading config file', config_path)
    with open(config_path, 'r') as in_file:
        model.input_length = json.load(in_file)['input_length']

    print('Reading tokenizer file', tokenizer_path)
    with open(tokenizer_path, 'rb') as in_file:
        model.tokenizer = pickle.load(in_file)

    print('Reading keras model file', model_path)
    model.model = keras.models.load_model(model_path)

    return model


def get_model_paths(directory):
    config_path = os.path.join(directory, 'config.json')
    tokenizer_path = os.path.join(directory, 'tokenizer.pickle')
    model_path = os.path.join(directory, 'keras-model')

    return config_path, tokenizer_path, model_path
