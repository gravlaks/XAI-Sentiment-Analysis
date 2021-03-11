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

# Local Application Modules
# -----------------------------------------------------------------------------------------
from emb import get_embeddings_index


def display_pca_scatter_plot(glove, words=["king", "queen", "tv", "radio", "boy", "girl"]):

    glove_idx = get_embeddings_index(glove)
    word_vectors = [glove_idx[word] for word in words]

    # For 2D, change rank to 2
    rank = 3

    three_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:rank]

    data = []
    count = 0

    trace = go.Scatter3d(
            x = three_dim[:,0],
            y = three_dim[:,1],
            z = three_dim[:,2],
            text = words,
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


    plot_figure = go.Figure(data = trace, layout = layout)
    plot_figure.show(block=True)

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-g', '--glove', required=True,  help=('The glove, pretrained embeddings, Leave out to perform a dry-run t'))

    args = parser.parse_args()
    
    display_pca_scatter_plot(args.glove)
   
