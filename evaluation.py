import matplotlib.pyplot as plt


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
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()
