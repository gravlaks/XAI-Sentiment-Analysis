# -*- coding: utf8 -*-

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
from keras.preprocessing.text import Tokenizer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

from emb import get_keras_embedding_layer

# Local Application Modules
# -----------------------------------------------------------------------------------------


def sequential_model(emb_layer):
    return tf.keras.Sequential([
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


def lstm_model(emb_layer):
    inp = tf.keras.Input(shape=(None,), name="title")
    embedded = emb_layer(inp)
    lstm = tf.keras.layers.LSTM(128)(embedded)
    flatten = tf.keras.layers.Flatten()(lstm)
    dense = tf.keras.layers.Dense(
        n_classes, activation='softmax')(flatten)
    return tf.keras.Model(
        inputs=inp,
        outputs=dense
    )


n_classes = 2
models = {
    'sequential': sequential_model,
    'recurrent': lstm_model,
}


class KerasTextClassifier():
    '''Wrapper class for keras text classification models that takes raw text as input.'''

    def __init__(self, max_words=30000, input_length=100):
        EMB_MAX_WORDS = 5000

        self.tokenizer = Tokenizer(
            num_words=EMB_MAX_WORDS, lower=False, split=' ', oov_token="UNK")

        self.input_length = input_length

    def init_model(self, emb_layer, model_type):
        self.model = self.get_model(emb_layer, model_type)

    def get_model(self, emb_layer, model_type):
        if model_type not in models:
            print(model_type, 'is not a valid model type. Possible values are:',
                  ', '.join(models.keys()))
        model = models[model_type](emb_layer)

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

        return self.model.fit(seqs, y_train, validation_data=(seqs_val, validation_data[1]),
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=verbose)

    def predict_proba(self, X, y=None, verbose=False):

        seqs = self._get_sequences(X)

        return self.model.predict(seqs)

    def evaluate(self, X, y=None, verbose=False):
        seqs = self._get_sequences(X)
        return self.model.evaluate(seqs, y, verbose=verbose)

    def save(self, directory, overwrite=False):
        save_classifier(self, directory, overwrite)


def create_classifier(glove_file, data, model_type):

    text_classifier = KerasTextClassifier()

    tokenizer = text_classifier.tokenizer
    emb_layer = get_keras_embedding_layer(glove_file, data['tweet'], tokenizer)
    text_classifier.init_model(emb_layer, model_type)
    model_path = './models/untrained'

    return text_classifier


def load_classifier(model_path):

    text_classifier_path = ('./classifiers/'
                            + model_path.split("/")[-1] + ".pkl")
    if not os.path.isfile(text_classifier_path):
        raise Exception("No text classifier object, create new model first")
    else:
        with open(text_classifier_path, 'rb') as in_file:
            text_classifier = pickle.load(in_file)

    text_classifier.model = tf.keras.models.load_model(model_path)
    text_classifier.model.layers[0].trainable = False

    return text_classifier


def save_classifier(text_classifier, model_path):

    # Cannot pickle tensorflow model, save it separately
    tf.keras.models.save_model(text_classifier.model, model_path)

    # Remove model from instance
    model_cache = text_classifier.model
    text_classifier.model = None

    # Pickle instance
    text_classifier_path = ('./classifiers/'
                            + model_path.split("/")[-1] + ".pkl")
    if not os.path.isdir('./classifiers'):
        os.mkdir('./classifiers')
    with open(text_classifier_path, 'wb') as out_file:
        print(text_classifier_path)
        pickle.dump(text_classifier, out_file, pickle.HIGHEST_PROTOCOL)

    # Restore old model object on instance
    text_classifier.model = model_cache
