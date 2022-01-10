#!/usr/bin/env python3
# -*- coding: utf8 -*-
# %%
import pickle


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from evaluation import evaluate_model, plot_history
from parse import load_data
from pre import preprocess, load_unprocessed, split_and_preprocess
# from explainability import explain_and_save, save_predictions, display_html_browser

from TextClassifierModel import load_classifier, new_classifier, save_classifier
# from visualize_embeddings import display_pca_scatter_plot


import codecs

# TODO set to True
PREPROCESS = False  # Do a fresh preprocess
EMB_MAX_WORDS = 8888
EMB_MAX_SEQUENCE_LENGTH = 30
RANDOM_SEED = 456

LOAD_TRAINED_MODEL = False
NEW_MODEL = True
# "sequential" | "recurrent"
MODEL_TYPE = "recurrent"
BATCH_SIZE = 64
EPOCHS = 2
SAVE_TRAINED_MODEL = True

PREPROCESS_INPUT = "./data/training.1600000.processed.noemoticon.csv"
PREPROCESS_OUTPUT = "./data/preprocessed.csv"
GLOVE_FILE = "./data/glove.6B.50d.txt"
MODEL_PKL = "./models/model.pkl"
TRAINED_MODEL_PATH = "models/trained_" + MODEL_TYPE
UNTRAINED_MODEL_PATH = "models/untrained_" + MODEL_TYPE
TWEETS_TO_EXPLAIN = [111, 156, 933487, 933565]


if PREPROCESS:
    preprocess(i=PREPROCESS_INPUT, o=PREPROCESS_OUTPUT)

data = load_data(PREPROCESS_OUTPUT)
X = data["tweet"]
y = data["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=RANDOM_SEED
)

y_train = tf.keras.utils.to_categorical(y_train, 2)
y_test = tf.keras.utils.to_categorical(y_test, 2)
y_val = tf.keras.utils.to_categorical(y_val, 2)

if not LOAD_TRAINED_MODEL:

    # Create Empty model
    if NEW_MODEL:
        text_classifier = new_classifier(
            glove_file=GLOVE_FILE,
            data=data,
            model_type=MODEL_TYPE,
            emb_max_words=EMB_MAX_WORDS,
            emb_max_sequence_length=EMB_MAX_SEQUENCE_LENGTH,
        )
        save_classifier(text_classifier, TRAINED_MODEL_PATH)
    else:
        text_classifier = load_classifier(model_path=UNTRAINED_MODEL_PATH)
    print(text_classifier.model.summary())

    history = text_classifier.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=1,
    )
    plot_history(history)
    if SAVE_TRAINED_MODEL:
        save_classifier(text_classifier, TRAINED_MODEL_PATH)
else:
    text_classifier = load_classifier(model_path=TRAINED_MODEL_PATH)

evaluate_model(text_classifier, X_test, y_test, verbose=False)


# df_orig = load_unprocessed(PREPROCESS_INPUT)
# explain_tweets_orig, explain_tweets_prep = split_and_preprocess(
#     df_orig, TWEETS_TO_EXPLAIN
# )
# explain_and_save(
#     explain_tweets_orig,
#     explain_tweets_prep,
#     TWEETS_TO_EXPLAIN,
#     text_classifier,
#     MODEL_TYPE,
# )


# save_predictions(explain_tweets_prep, TWEETS_TO_EXPLAIN, text_classifier)

# html = codecs.open(
#     f"data/predictions/html_{TWEETS_TO_EXPLAIN[0]}.html", "r").read()
# display_html_browser(html, "explainability")
# display_pca_scatter_plot(GLOVE_FILE)
