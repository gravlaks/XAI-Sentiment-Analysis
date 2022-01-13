# -*- coding: utf8 -*-

import matplotlib.pyplot as plt
import os


def evaluate_model(model, X_test, y_test, verbose=False):
    # The labels here assume we use accuracy for our model
    score = model.evaluate(X_test, y_test, verbose=verbose)
    print('Accuracy', score[1])


def plot_history(history):
    # The labels here assume we use accuracy for our model
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    filename = 'output/model_accuracy.png'
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)

    plt.clf()  # clear figure
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # plt.show()
    plt.savefig('output/model_loss.png')
