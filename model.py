from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from keras.layers import Input, Embedding, Lambda, Bidirectional, LSTM, Dense, Dropout, Flatten, Activation
from keras.models import Model
import numpy as np

# TODO: TEMP vars, delete when pipeline is ready for it
maxlen = 30
vocab_size = 1000
embedding_matrix = np.zeros((1000, 100))


def build_model_keras():
    model = Sequential()
    model.add(Embedding(vocab_size, 100, weights=[
        embedding_matrix], trainable=False, name='GloVe_Embedding'))
    model.add(LSTM(128, return_sequences=True))
    # model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)))
    # model.add(Dropout(0.05))
    # model.add(LSTM(16))
    model.add(Dense(1, activation='relu',
                    kernel_regularizer=regularizers.l2(0.01)))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy', metrics=['acc'])

    return model
