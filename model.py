#!/bin/bash
"""
File name: model.py

Creation Date: Wed 10 Mar 2021

Description:

"""

# Python Libraries
# -----------------------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin
import tensorflow as tf

# Local Application Modules
# -----------------------------------------------------------------------------------------
from emb import get_keras_embeddings_layer

class KerasTextClassifier():
    '''Wrapper class for keras text classification models that takes raw text as input.'''
    
    def __init__(self,glove, prepoc, tokenizer, max_words=30000, input_length=100, n_classes=3, epochs=10, batch_size=4000):
        self.glove = glove
        self.prepoc = prepoc
        self.input_length = input_length
        self.n_classes = n_classes
        self.epochs = epochs
        self.bs = batch_size
        self.model = self._get_model()
        self.tokenizer = tokenizer
        
    
    def _get_model(self):
        
        emb_layer = get_keras_embeddings_layer(self.glove, self.prepoc)
        sequential_model = tf.keras.Sequential([
            emb_layer,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dense(100, activation='sigmoid'),
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])

        inp = Input(shape=(None,), name="title"
                )
        embedded = emb_layer(inp)
        lstm = tf.keras.layers.LSTM(128)(embedded)
        flatten = tf.keras.layers.Flatten()(lstm)
        dense = tf.keras.layers.Dense(self.n_classes, activation='sigmoid')(flatten)
        lstm_model = tf.keras.Model(
                inputs=inp,
                outputs=dense
        )



        model = sequential_model

        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        print(model.summary())
        return model
    
    def _get_sequences(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.input_length, value=0)
    
    
    def fit(self, X, y):
        '''
        Fit the vocabulary and the model.
        
        :params:
        X: list of texts.
        y: labels.
        '''
        print("Fit")
        print(X)
        

        #self.tokenizer.fit_on_texts(X)
        seqs = self._get_sequences(X)
        print("Fit")
        print(seqs)
        
        print("Fit ys")
        print(y)
        

        self.model.fit(seqs, y, batch_size=self.bs, epochs=self.epochs, validation_split=0.1)
    
    def predict_proba(self, X, y=None):
        
        seqs = self._get_sequences(X) 

        return self.model.predict(seqs)
    
    """
    def predict(self, X, y=None):
        return np.argmax(self.predict_proba(X), axis=1)
    """



def build_model_keras(prepoc, glove, tokenizer):
    model = KerasTextClassifier(glove, prepoc,epochs=1, input_length=100, tokenizer)

    return model
