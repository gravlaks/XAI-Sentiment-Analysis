from keras.layers import Input, Embedding, Lambda, Bidirectional, LSTM, Dense, Dropout, Flatten, Activation
from keras.models import Model


maxlen = 30
vocab_size = 1000
embedding_matrix = [[0 for _ in range(100)]]

input_layer = Input(shape=(maxlen,), name='Input')
embedding_layer = Embedding(vocab_size, 100, weights=[
                            embedding_matrix], trainable=False, name='GloVe_Embedding')(input_layer)
LSTM_layer = LSTM(128)(embedding_layer)
dropout_layer = Dropout(0.25)(LSTM_layer)
# We have 7 output labels, so units=7
dense_layer_1 = Dense(units=7, activation='sigmoid')(dropout_layer)


model = Model(inputs=[input_layer], outputs=[dense_layer_1])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())
