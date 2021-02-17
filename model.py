import csv
import numpy as np
from IPython.core.display import display
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from eli5 import show_weights
from eli5 import show_prediction
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

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

    for x in all_xs:
        if x['Age']:
            x['Age'] = float(x['Age'])
        else:
            x.pop('Age')
        x['Fare'] = float(x['Fare'])
        x['SibSp'] = int(x['SibSp'])
        x['Parch'] = int(x['Parch'])



    clf = XGBClassifier()
    vec = DictVectorizer()
    pipeline = make_pipeline(vec, clf)

    def evaluate(_clf):
        scores = cross_val_score(_clf, all_xs, all_ys, scoring='accuracy', cv=10)
        print('Accuracy: {:.3f} ± {:.3f}'.format(np.mean(scores), 2 * np.std(scores)))
        _clf.fit(train_xs, train_ys)  # so that parts of the original pipeline are fitted

    evaluate(pipeline)

    booster = clf.get_booster()
    original_feature_names = booster.feature_names
    booster.feature_names = vec.get_feature_names()
    print(booster.get_dump()[0])
    # recover original feature names
    booster.feature_names = original_feature_names

    show_weights(clf, vec=vec)
    show_prediction(clf, valid_xs[1], vec=vec, show_feature_values=True)

    no_missing = lambda feature_name, feature_value: not np.isnan(feature_value)
    show_prediction(clf, valid_xs[1], vec=vec, show_feature_values=True, feature_filter=no_missing)

    vec2 = FeatureUnion([
        ('Name', CountVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 4),
            preprocessor=lambda x: x['Name'],
            max_features=100,
        )),
        ('All', DictVectorizer()),
    ])
    clf2 = XGBClassifier()
    pipeline2 = make_pipeline(vec2, clf2)
    evaluate(pipeline2)

    show_weights(clf2, vec=vec2)

    for idx in [4, 5, 7, 37, 81]:
        display(show_prediction(clf2, valid_xs[idx], vec=vec2,
                                show_feature_values=True, feature_filter=no_missing))