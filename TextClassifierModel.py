#!/bin/bash
"""
File name: model.py

Creation Date: Wed 10 Mar 2021

Description:

"""

# Python Libraries
# -----------------------------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np


# Local Application Modules
# -----------------------------------------------------------------------------------------


class KerasTextClassifier():
    '''Wrapper class for keras text classification models that takes raw text as input.'''

    def __init__(self, tokenizer, emb_layer, max_words=30000, input_length=100):
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


def build_model_keras(tokenizer, emb_layer):
    print("building model")
    model = KerasTextClassifier(tokenizer, emb_layer)

    return model
