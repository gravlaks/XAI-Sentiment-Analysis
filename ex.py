#!/usr/bin/env python3
# -*- coding: utf8 -*-

import pickle

import eli5
import numpy as np
import pandas as pd
import tensorflow as tf
from eli5.lime import TextExplainer
from sklearn.model_selection import train_test_split

from evaluation import evaluate_model, plot_history
from parse import load_data
from pre import preprocess
from TextClassifierModel import (load_classifier, new_classifier,
                                 save_classifier)
from visualize_embeddings import display_pca_scatter_plot

PREPROCESS = True  # Do a fresh preprocess
MAKE_NEW_EMBEDDING = True  # If False, the stored one will be loaded
# EMB_MAX_WORDS = None
RANDOM_SEED = 456

NEW_MODEL = True
SAVE_TRAINED_MODEL = True

PREPROCESS_INPUT = './data/training.1600000.processed.noemoticon.csv'
PREPROCESS_OUTPUT = './data/preprocessed.csv'
GLOVE_FILE = './data/glove.6B.50d.txt'
EMB_PKL = './models/emb_layer.pkl'
MODEL_PKL = './models/model.pkl'


if PREPROCESS:
    preprocess(i=PREPROCESS_INPUT, o=PREPROCESS_OUTPUT)


data = load_data(PREPROCESS_OUTPUT)
X = data['tweet']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_SEED)

y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)
y_val = tf.keras.utils.to_categorical(y_val, 2)


# Create Empty model

trained_model_path = 'models/trained'
untrained_model_path = 'models/untrained'

if NEW_MODEL:
    model_type = "recurrent"
    text_classifier = new_classifier(
        glove_file=GLOVE_FILE, data=data, model_type=model_type)
    save_classifier(text_classifier, 'models/untrained')
else:
    text_classifier = load_classifier(model_path=untrained_model_path)
print(text_classifier.model.summary())


history = text_classifier.fit(X_train, y_train, validation_data=(
    X_val, y_val), batch_size=60, epochs=30, verbose=1)
plot_history(history)


save_classifier(text_classifier, 'models/trained')


evaluate_model(text_classifier, X_test, y_test, verbose=False)


te = TextExplainer(random_state=42)
te.fit("I love candy. I like to be positive, be happy! What a lovely day",
       text_classifier.predict_proba)
te.show_prediction()

display_pca_scatter_plot(GLOVE_FILE)
