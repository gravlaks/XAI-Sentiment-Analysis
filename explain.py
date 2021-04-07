#!/usr/bin/env python3
# -*- coding: utf8 -*-

import pickle

import eli5
import numpy as np
import pandas as pd
import tensorflow as tf
from eli5.lime import TextExplainer
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

from evaluation import evaluate_model
from parse import load_data
from TextClassifierModel import new_classifier


# Based on https://xkcd.com/221/
def generate_random_seed():
    # Chosen by three fair dice rolls.
    # Guaranteed to be random.
    return 456


RANDOM_SEED = generate_random_seed()


def main(args):
    in_file = args.input
    in_glove = args.glove
    model_type = args.type

    print('Loading data')
    data = load_data(in_file)

    print('Creating tokenizer')
    tokenizer = Tokenizer(num_words=5000, lower=True,
                          split=' ', oov_token="UNK")

    print('Loading training data')
    data = load_data(in_file)
    X = data['tweet']
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_SEED)

    y_train = tf.keras.utils.to_categorical(y_train, 2)
    y_test = tf.keras.utils.to_categorical(y_test, 2)
    y_val = tf.keras.utils.to_categorical(y_val, 2)

    print('Creating model')
    model = new_classifier(glove_file=in_glove,
                           data=data,
                           model_type=model_type)
    for layer in model.model.layers:
        if isinstance(layer, tf.keras.layers.Embedding):
            layer.trainable = False

    print('Fitting model')
    model.fit(X_train, y_train, validation_data=(X_val, y_val),
              batch_size=60,
              epochs=30,
              verbose=1)

    evaluate_model(model, X_test, y_test, verbose=False)

    print('Explaining')
    te = TextExplainer(random_state=42)
    texts = [
        "I love candy. I like to be positive, be happy! What a lovely day",
        "I hate you shitty fuckers"
    ]
    for text in texts:
        print()
        print('-' * 50)
        print('PREDICTION FOR:    ', text)
        te.fit(text, model.predict_proba)
        ex = te.explain_prediction()
        print(eli5.format_as_text(ex))
        print('-' * 50)
        print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', required=True,
                        help='The preprocessed file from running pre.py')
    parser.add_argument('-g', '--glove', default='./data/glove.6B.50d.txt',
                        help='The glove file')
    parser.add_argument('-t', '--type', default='sequential',
                        help='Model type, can be either "sequential" or "recurrent"')
    args = parser.parse_args()

    main(args)
