#!/bin/bash
"""
File name: visualize_embeddings.py

Creation Date: Wed 10 Mar 2021

Description:

"""

# Python Libraries
# -----------------------------------------------------------------------------------------

from sklearn.decomposition import PCA
import plotly
import numpy as np
import plotly.graph_objs as go
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

# Local Application Modules
# -----------------------------------------------------------------------------------------
from emb import get_embeddings_index

def find_similar_words(glove_idx, word):
    
    word_vec = glove_idx[word]
    
    dists = np.array((len(glove_idx), ))

    idx = 0
    for word, vec in glove_idx.items():
        if idx == 10:
            break

        dist = np.linalg.norm(vec-word_vec)
        print(dist)
        dists[idx] = dist

    


def display_pca_scatter_plot(glove):
    words = ["king", "queen", "knight", "palace", "castle",  "tv", "radio", "television", "video", "channel", "boy", "girl", "child", "teenager", "teen"]

    glove_idx = get_embeddings_index(glove)

    # For 2D, change rank to 2
    rank = 3
    word_vectors = [glove_idx[word] for word in words]
    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:rank]
    data = []
    count = 0

    for i in range(0, len(words), 5):



        trace = go.Scatter3d(
                x = three_dim[i:i+5,0],
                y = three_dim[i:i+5,1],
                z = three_dim[i:i+5,2],
                text = words[i:i+5],
            name='random name',
            textposition = "top center",
            textfont_size = 20,
            mode = 'markers+text',
            marker = {
                'size': 10,
                'opacity': 0.8,
                'color': 2
            }

        )
        data.append(trace)



# Configure the layout

    layout = go.Layout(
        margin = {'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
        x=1,
        y=0.5,
        font=dict(
            family="Courier New",
            size=25,
            color="black"
        )),
        font = dict(
            family = " Courier New ",
            size = 15),
        autosize = False,
        width = 1000,
        height = 1000
        )


    plot_figure = go.Figure(data = data, layout = layout)
    plot_figure.show(block=True)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--glove', required=True,  help=('The glove, pretrained embeddings, Leave out to perform a dry-run t'))

    args = parser.parse_args()
    
    display_pca_scatter_plot(args.glove)
   
