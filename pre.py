#!/usr/bin/env python3
# -*- coding: utf8 -*-

import csv
import os
import re
from json import dumps

import nltk
import pandas as pd
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from tqdm import tqdm


def preprocess(i, o, slice=None):
    # parse
    print('Loading', i)
    df = pd.read_csv(i, header=None)
    if slice:
        df = df.sample(n=int(slice))

    # drop less important columns
    print('Stripping down')
    df = df[[0, 5]]
    df.columns = ['target', 'tweet']

    # apply preprocessing steps per row
    print('Preprocessing')
    with tqdm(total=len(df)) as progress_bar:
        df['tweet'] = df['tweet'].apply(
            lambda tweet: preprocess_row(tweet, progress_bar))

    # Convert target 0->0 and 4->1
    df['target'] = df['target'].apply(target_mapping)

    # write out if not dry-run
    if o is not None:
        print('Done.', 'Writing to', o)
        df.to_csv(o, index=False)
    else:
        print('Dry-run: not writing to disk.')
        print(df)

    print('Success!')


# read in custom stopwords.txt and strip away apostrophes correctly
stopwords = set()
with open('stopwords.txt') as csvfile:
    lines = csv.reader(csvfile, delimiter=',')
    for row in lines:
        for word in row:
            if len(word) and word[-1] != "'" and word[0] != "'":
                stopwords.add(word)
            elif len(word):
                stopwords.add(word[:-1][1:])

regex_user = re.compile(r'@[a-zæøåäöüß]+\d*')
regex_cashtag = re.compile(r'\$([a-zæøåäöüß._]+|\d+\w+_\w+)')
regex_URL = re.compile(
    r'(http|ftp|https)(:\/\/)([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
regex_digit = re.compile(r'\d')
regex_punctuation_non_numeral = re.compile(
    r'[\.,<>!?:;*\-\^\']|\&[gl]t;(?!\d)')
regex_amp = re.compile(r'\&amp;')


def preprocess_row(tweet, progress_bar):
    # transform to lowercase
    tweet = tweet.lower()
    # remove a bunch of things
    tweet = re.sub(regex_user, 'ID', tweet)
    tweet = re.sub(regex_cashtag, 'TICKER', tweet)
    tweet = re.sub(regex_URL, 'URL', tweet)
    tweet = re.sub(regex_punctuation_non_numeral, '', tweet)
    tweet = re.sub(regex_digit, 'D', tweet)
    tweet = re.sub(regex_amp, '&', tweet)
    # only keep words in hashtags
    tweet = tweet.replace('#', ' ')

    # remove stopwords
    words = word_tokenize(tweet)
    words = [lemmatize(word)
             for word in words
             if word not in stopwords]

    progress_bar.update(1)
    return dumps(words)


lemmatizer = WordNetLemmatizer()


def lemmatize(word):
    return lemmatizer.lemmatize(word, get_wordnet_pos(word))


def get_wordnet_pos(word):
    # Stolen from https://www.machinelearningplus.com/nlp/lemmatization-examples-python/
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def target_mapping(n):
    if n == 4:
        return 1
    return n


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-i', '--input', default='./data/training.1600000.processed.noemoticon.csv',
                        help='The file to preprocess')
    parser.add_argument('-o', '--output',
                        help=('The output file to write the preprocessed data to. '
                              'Leave out to perform a dry-run that does not write to disk.'))
    parser.add_argument('-s', '--slice',
                        help=('Number of rows to include'))

    args = parser.parse_args()

    if not args.output:
        print('No output location specified, performing a dry-run')

    preprocess(args.input, args.output, args.slice)


def load_unprocessed(i):
    df_orig = pd.read_csv(i, header=None)
    df_orig = df_orig[[0, 5]]

    df_orig.columns = ['target', 'tweet']
    return df_orig

def split_and_preprocess(df_orig, indices):

    df_sliced = df_orig.loc[indices]
    df_prep = df_sliced.copy()

    #preprocess
    with tqdm(total=len(indices)) as progress_bar:
        df_prep['tweet'] = df_prep['tweet'].apply(
            lambda tweet: preprocess_row(tweet, progress_bar))

    explain_tweets_orig = df_sliced['tweet'].copy()

    explain_tweets_prep = df_prep['tweet']
    return explain_tweets_orig, explain_tweets_prep