#!/usr/bin/env python3
# -*- coding: utf8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf
import pickle

PREPROCESS = True # Do a fresh preprocess
MAKE_NEW_EMBEDDING = True # If False, the stored one will be loaded
# EMB_MAX_WORDS = None
RANDOM_SEED = 456
SAVE_TRAINED_MODEL = True

PREPROCESS_INPUT = './data/training.1600000.processed.noemoticon.csv'
PREPROCESS_OUTPUT = './data/preprocessed.csv'
GLOVE_FILE = './data/glove.6B.50d.txt'
EMB_PKL = './models/emb_layer.pkl'
MODEL_PKL = './models/model.pkl'

from pre import preprocess

if PREPROCESS:
    preprocess(i=PREPROCESS_INPUT, o=PREPROCESS_OUTPUT)

from parse import load_data
from sklearn.model_selection import train_test_split

data = load_data(PREPROCESS_OUTPUT)
X = data['tweet']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=RANDOM_SEED)

y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)
y_val = tf.keras.utils.to_categorical(y_val, 2)


### Create Empty model
from TextClassifierModel import new_classifier, load_classifier, save_classifier




trained_model_path = 'models/trained'
untrained_model_path = 'models/untrained'

new_model = True
if new_model:
    model_type = "recurrent"
    text_classifier = new_classifier(glove_file = GLOVE_FILE, data=data, model_type=model_type)
    save_classifier(text_classifier, 'models/untrained')
else:
    text_classifier = load_classifier(model_path=untrained_model_path)
print(text_classifier.model.summary())

from evaluation import plot_history

history = text_classifier.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=60, epochs=30, verbose=1)
plot_history(history)

from TextClassifierModel import save_classifier
save_classifier(text_classifier, 'models/trained')

from evaluation import evaluate_model

evaluate_model(text_classifier, X_test, y_test, verbose=False)

import eli5
from eli5.lime import TextExplainer

te = TextExplainer(random_state=42)
te.fit("I love candy. I like to be positive, be happy! What a lovely day", text_classifier.predict_proba)
te.show_prediction()

