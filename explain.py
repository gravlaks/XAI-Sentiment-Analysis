#!/usr/bin/env python3
# -*- coding: utf8 -*-

import pickle

import eli5
import numpy as np
import pandas as pd
import tensorflow as tf
from eli5.lime import TextExplainer
from keras.preprocessing.text import Tokenizer

from emb import get_keras_embeddings_layer
from parse import load_data
from TextClassifierModel import build_model_keras


def load_training_data(model, data):
    def mapping(n):
        if n == 0:
            return 0
        if n == 4:
            return 1

    data_0 = data[data['target'] == 0]
    data_4 = data[data['target'] == 4]

    count_0, count_4 = data.target.value_counts()
    print(count_0, count_4)

    split = len(data['tweet'])//2

    training_data = np.array(data['tweet'][:split])
    training_target = np.array(data['target'][:split])
    test_data = np.array(data['tweet'][split:])
    test_target = np.array(data['target'][split:])

    for i in range(len(training_target)):
        training_target[i] = mapping(training_target[i])
    for i in range(len(test_target)):
        test_target[i] = mapping(test_target[i])
    training_target = tf.keras.utils.to_categorical(training_target, 2)
    test_target = tf.keras.utils.to_categorical(test_target, 2)

    return training_data, training_target, test_data, test_target


def main(args):
    in_file = args.input
    in_glove = args.glove
    model_type = args.type

    print('Loading data')
    data = load_data(in_file)

    print('Creating tokenizer, embedding, model')
    tokenizer = Tokenizer(num_words=5000, lower=True,
                          split=' ', oov_token="UNK")
    emb_layer = get_keras_embeddings_layer(
        in_glove, in_file, tokenizer)
    model = build_model_keras(tokenizer, emb_layer, model_type)

    print('Loading training data')
    training_data, training_target, _, _ = load_training_data(
        model, data
    )

    print('Fitting model')
    model.fit(training_data, training_target, epochs=40, batch_size=30)

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
                        help='Model type, can be either sequential or recurrent')
    args = parser.parse_args()

    main(args)
