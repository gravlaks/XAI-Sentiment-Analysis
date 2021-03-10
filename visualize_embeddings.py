#!/bin/bash
"""
File name: visualize_embeddings.py

Creation Date: Wed 10 Mar 2021

Description:

"""

# Standard Python Libraries
# -----------------------------------------------------------------------------------------

# Local Application Modules
# -----------------------------------------------------------------------------------------

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-p', '--prepoc', default='./data/prepoc.csv',
                        help='The file to preprocess')
    parser.add_argument('-g', '--glove',
                        help=('The glove, pretrained embeddings'
                              'Leave out to perform a dry-run t))

    args = parser.parse_args()
