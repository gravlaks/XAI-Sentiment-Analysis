#!/usr/bin/env python3
# -*- coding: utf8 -*-

"""
File name: visualize_embeddings.py

Creation Date: Wed 10 Mar 2021

Description: Visualizes embeddings by using PCA dimension reduction, heavily inspired by 
https://towardsdatascience.com/visualizing-word-embedding-with-pca-and-t-sne-961a692509f5

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
            name=f"Class {i//5+1}",
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
    axes_range = [-5, 5]
    plot_figure.update_layout(
    scene = dict(
        xaxis = dict(nticks=4, range=axes_range,),
                     yaxis = dict(nticks=4, range=axes_range,),
                     zaxis = dict(nticks=4, range=axes_range,),),
    margin=dict(r=20, l=10, b=10, t=10))
    plot_figure.show(block=True)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--glove', required=True,  help=('The glove, pretrained embeddings, Leave out to perform a dry-run t'))

    args = parser.parse_args()
    
    display_pca_scatter_plot(args.glove)
   
