#!./venv/bin/python3
# -*- coding: utf8 -*-


import os
import re

import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords


def preprocess(i, o):
    # parse
    print('Loading', i)
    df = pd.read_csv(i, header=None)

    # drop less important columns
    print('Stripping down')
    df = df[[0, 5]]
    df.columns = ['target', 'tweet']

    # apply preprocessing steps per row
    print('Preprocessing')
    df['tweet'] = df['tweet'].apply(preprocess_row)

    # write out if not dry-run
    if o is not None:
        print('Done.', o)
        df.to_csv(o, index=False)
    else:
        print('Dry-run: not writing to disk.')
        print(df)

    print('Success!')


stopword_set = set(stopwords.words('english'))

regex_user = re.compile(r'@[a-zæøåäöüß]+\d*')
regex_cashtag = re.compile(r'\$([a-zæøåäöüß._]+|\d+\w+_\w+)')
regex_URL = re.compile(
    r'(http|ftp|https)(:\/\/)([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
regex_digit = re.compile(r'\d')
regex_punctuation_non_numeral = re.compile(
    r'(?<!\d)[\.,<>!?:\-\^]|\&[gl]t;(?!\d)')
regex_amp = re.compile(r'\&amp;')


def preprocess_row(tweet):
    # transform to lowercase
    tweet = tweet.lower()
    # remove a bunch of things
    tweet = re.sub(regex_user, 'ID', tweet)
    tweet = re.sub(regex_cashtag, 'TICKER', tweet)
    tweet = re.sub(regex_URL, 'URL', tweet)
    tweet = re.sub(regex_punctuation_non_numeral, '', tweet)
    tweet = re.sub(regex_digit, 'D', tweet)
    tweet = re.sub(regex_amp, '&', tweet)
    # only keeps words in hashtags
    tweet = tweet.replace('#', ' ')

    # remove stopwords
    tokens = word_tokenize(tweet)
    tokens = [token for token in tokens if token not in stopword_set]

    return tokens


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', default='./data/training.1600000.processed.noemoticon.csv',
                        help='The file to preprocess')
    parser.add_argument('-o', '--output',
                        help=('The output file to write the preprocessed data to. '
                              'Leave out to perform a dry-run that does not write to disk.'))

    args = parser.parse_args()

    if not args.output:
        print('No output location specified, performing a dry-run')

    preprocess(args.input, args.output)