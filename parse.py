#!./venv/bin/python3
# -*- coding: utf8 -*-

from json import loads

import pandas as pd


def load_data(path):
    df = pd.read_csv(path)
    df['tweet'] = df['tweet'].apply(loads)
    return df
