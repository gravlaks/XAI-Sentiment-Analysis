#!./venv/bin/python3
# -*- coding: utf8 -*-

import os
import pandas as pd


def preprocess(i, o):

    # parse
    print('Loading', i)
    df = pd.read_csv(i, header=None)

    # drop less important columns
    print('Stripping down')
    df = df[[0, 5]]

    # write out if not dry-run
    if o is not None:
        print('Done.', o)
        df.to_csv(o, index=False)
    else:
        print('Dry-run: not writing to disk.')
        print(df)

    print('Success!')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('input',
                        help='The file to preprocess')
    parser.add_argument('-o', '--output',
                        help=('The output file to write the preprocessed data to. '
                              'Leave out to perform a dry-run that does not write to disk.'))

    args = parser.parse_args()

    if not args.output:
        print('No output location specified, performing a dry-run')

    preprocess(args.input, args.output)

