import csv
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    with open('titanic/train.csv', 'rt') as f:
        data = list(csv.DictReader(f))
    data[:1]
    # print(data[:1])
    _all_xs = [{k: v for k, v in row.items() if k != 'Survived'} for row in data]
    _all_ys = np.array([int(row['Survived']) for row in data])

    all_xs, all_ys = shuffle(_all_xs, _all_ys, random_state=0)
    train_xs, valid_xs, train_ys, valid_ys = train_test_split(
        all_xs, all_ys, test_size=0.25, random_state=0)
    print('{} items total, {:.1%} true'.format(len(all_xs), np.mean(all_ys)))


