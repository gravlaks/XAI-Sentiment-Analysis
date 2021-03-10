"""
File name: emb.py

Creation Date: Mi 17 Feb 2021

Description: Embedding layer

"""

# Instructions
# -----------------------------------------------------------------------------
# Run emb.py with -m, -g and -p for model choice, glover file and processed file. Glover file used is Example file used from:https://www.kaggle.com/watts2/glove6b50dtxt  
# The second file is the preprocessed file we get from pre.py.
# Model choices are wither 0 for keras or 1 for xgboost. 

# Python Libraries
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
from keras.initializers import Constant
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import eli5
from eli5.lime import TextExplainer

# Keras model test
# -----------------------------------------------------------------------------
from time import time
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from keras.models import Model, Input
from sklearn.base import BaseEstimator, TransformerMixin
from keras.layers import Dense, LSTM, Dropout, Embedding, SpatialDropout1D, Bidirectional, concatenate
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D


strategy = tf.distribute.get_strategy()


# Local Application Modules
# -----------------------------------------------------------------------------
from parse import load_data

#var definition
MAX_WORDS = 5000
MAXLEN = 100


def get_embeddings_index(dat):
    embeddings_index = {}
    with open(dat, 'r',encoding="utf8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')

            embeddings_index[word] = coefs
    
    print("Glove data loaded")
    return embeddings_index

def get_tokenizer(prepoc):
    df = load_data(prepoc)
    df = pd.read_csv(prepoc, converters={'tweet': eval})
    tweets = df['tweet']

    #Tokenizer magic
    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(tweets)
    #temporary to test out the model 
    tokenized_tweets = tokenizer.texts_to_sequences(tweets)
    #remove this return sequence. for testing purposes. 
    return tokenizer,tokenized_tweets,df

##bread and butter of this whole operation. 
def get_embedding_matrix(glove, prepoc):
    #call the tokenizer
    word_idx = get_tokenizer(prepoc)[0].word_index 
    #Embeding bart :)
    embeddings_index = get_embeddings_index(glove)

    EMBEDDING_DIM = embeddings_index.get('a').shape[0]

    num_words = min(MAX_WORDS, len(word_idx))+1
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    not_found_count = 0
    for word, i in word_idx.items():
        if i > MAX_WORDS:
            continue
        embedding_vector = embeddings_index.get(word) ## This references the loaded embeddings dictionary
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            not_found_count += 1
    print(f"Words not found {not_found_count}")
    
    return  num_words, EMBEDDING_DIM, embedding_matrix

#Keras embedding layer.
def get_keras_embeddings_layer(glove, prepoc):
    num_words,EMBEDDING_DIM,embedding_matrix = get_embedding_matrix(glove, prepoc)
    MAX_SEQUENCE_LENGTH = 100
    
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    
    return  embedding_layer

##cgboost verison. sentence_feature_v2 returns the mean of the embedding matrix. This is the only thing necessary to create the training and test data. 
def sentence_features_v2(s, embedding_matrix,emb_size):
    M=np.array([embedding_matrix[w] for w in s])
    return M.mean(axis=0)

#test out if the embeding matrix works with xgboost. Simple model. 

class KerasTextClassifier(BaseEstimator, TransformerMixin):
    '''Wrapper class for keras text classification models that takes raw text as input.'''
    
    def __init__(self,glove, prepoc, max_words=30000, input_length=100, n_classes=3, epochs=10, batch_size=4000):
        self.glove = glove
        self.prepoc = prepoc
        self.input_length = input_length
        self.n_classes = n_classes
        self.epochs = epochs
        self.bs = batch_size
        self.model = self._get_model()
        self.tokenizer = Tokenizer(num_words=MAX_WORDS,
                                   lower=True, split=' ', oov_token="UNK")
    
    def _get_model(self):
        emb_layer = get_keras_embeddings_layer(self.glove, self.prepoc)
        sequential_model = tf.keras.Sequential([
            emb_layer,
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            tf.keras.layers.Dense(100, activation='sigmoid'),
            tf.keras.layers.Dense(self.n_classes, activation='softmax')
        ])

        inp = Input(shape=(None,), name="title"
                )
        embedded = emb_layer(inp)
        lstm = tf.keras.layers.LSTM(128)(embedded)
        flatten = tf.keras.layers.Flatten()(lstm)
        dense = tf.keras.layers.Dense(self.n_classes, activation='sigmoid')(flatten)
        lstm_model = tf.keras.Model(
                inputs=inp,
                outputs=dense
        )



        model = sequential_model

        model.compile(optimizer="adam",
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        print(model.summary())
        return model
    
    def _get_sequences(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.input_length, value=0)
    
    
    def fit(self, X, y):
        '''
        Fit the vocabulary and the model.
        
        :params:
        X: list of texts.
        y: labels.
        '''
        print("Fit")
        print(X)
        

        self.tokenizer.fit_on_texts(X)
        seqs = self._get_sequences(X)
        print("Fit")
        print(seqs)
        
        print("Fit ys")
        print(y)
        

        self.model.fit(seqs, y, batch_size=self.bs, epochs=self.epochs, validation_split=0.1)
    
    def predict_proba(self, X, y=None):
        
        print("Predict proba")
        print(X)
        seqs = self._get_sequences(X) 
        print("Predict proba")
        print(seqs)

        return self.model.predict(seqs)
    
    """
    def predict(self, X, y=None):
        return np.argmax(self.predict_proba(X), axis=1)
    """
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)




#Test out if keras is working, simple model. 







if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-m', '--model', default=0,
                        help='Choose model type to use, 0 for keras and 1 for xgboost. Default is Keras')
    parser.add_argument('-g', '--glove', default='./data/glove.6B.50d.txt',
                        help='The glover twitter file to train glover ')
    parser.add_argument('-p', '--prepoc',
                        help='The preprocessed file from running pre.py. ')

    args = parser.parse_args()

    if not args.glove or not args.prepoc:
        print('We need at least the preprocessed file boys. Try again with -p ***')
        exit()
    elif(int(args.model) == 1):
        xgboost_model_test(args.glove, args.prepoc)
    else:
        data = pd.read_csv(args.prepoc, converters={'tweet': eval})

        data_0 = data[data['target']==0]
        data_4 = data[data['target']==4]

        count_0, count_4 = data.target.value_counts()
        print(count_0, count_4)



        split = len(data['tweet'])//2

        training_data = np.array(data['tweet'][:split])
        target_data = np.array(data['target'][:split])




        def mapping(n):
            if n == 0:
                return 0
            if n == 2:
                print("neutral")
                return 1
            
            if n == 4:
                return 2
        

        for i in range(len(target_data)):
            target_data[i] = mapping(target_data[i])
        #print("Target data")
        #print(target_data)
        target_data = tf.keras.utils.to_categorical(target_data, 3)

        doc = np.array(data['tweet'][:1])
        
        




        text_model = KerasTextClassifier(args.glove, args.prepoc,epochs=1, input_length=100)
        text_model.fit(training_data, target_data)

        pred = text_model.predict_proba(doc)
        print(pred.shape)
        
        
        
       
        
        te = TextExplainer(random_state=42)
        te.fit("I kill and murder. I hate you. A very violent tweet, violence damn fuck", text_model.predict_proba)
        html = te.show_prediction().data
        print(type(html))

        with open("data/data.html", "w") as file:
            file.write(html)




