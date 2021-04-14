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

# TODO set to True
PREPROCESS = False  # Do a fresh preprocess
EMB_MAX_WORDS = 8888
RANDOM_SEED = 456

LOAD_TRAINED_MODEL = False
NEW_MODEL = True
# "sequential" | "recurrent"
MODEL_TYPE = "sequential"
BATCH_SIZE = 64
EPOCHS = 1
SAVE_TRAINED_MODEL = False

PREPROCESS_INPUT = './data/training.1600000.processed.noemoticon.csv'
PREPROCESS_OUTPUT = './data/preprocessed.csv'
GLOVE_FILE = './data/glove.6B.50d.txt'
MODEL_PKL = './models/model.pkl'
TRAINED_MODEL_PATH = 'models/trained_'+MODEL_TYPE
UNTRAINED_MODEL_PATH = 'models/untrained_'+MODEL_TYPE

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

if not LOAD_TRAINED_MODEL:

    # Create Empty model
    if NEW_MODEL:
        text_classifier = new_classifier(
            glove_file=GLOVE_FILE, data=data, model_type=MODEL_TYPE)
        save_classifier(text_classifier, TRAINED_MODEL_PATH)
    else:
        text_classifier = load_classifier(
            model_path=UNTRAINED_MODEL_PATH)
    print(text_classifier.model.summary())

    history = text_classifier.fit(X_train, y_train, validation_data=(
        X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
    plot_history(history)
    if SAVE_TRAINED_MODEL:
        save_classifier(text_classifier, TRAINED_MODEL_PATH, MODEL_TYPE)
else:
    text_classifier = load_classifier(
        model_path=TRAINED_MODEL_PATH)

evaluate_model(text_classifier, X_test, y_test, verbose=False)


te = TextExplainer(random_state=42)
te.fit("I love candy. I like to be positive, be happy! What a lovely day",
       text_classifier.predict_proba)
te.show_prediction()

display_pca_scatter_plot(GLOVE_FILE)
