import os

from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesRegressor

import numpy as np


class MyPotentialTransformer:

    def fit(self, x, y):
        return self

    def fit_transform(self, x, y):
        return self.transform(x)

    def transform(self, x):
        ans = []
        for xi in x:
            i, j = np.where(xi == np.min(xi))
            x_shift, y_shift = round(np.mean(i)), round(np.mean(j))
            temp = np.roll(xi, 128 - x_shift, axis=0)
            new = np.roll(temp, 128 - y_shift, axis=1)
            ans.append(np.asarray(new).reshape(1, -1).ravel())
        return np.asarray(ans)


def load_dataset(data_dir):
    """
    Read potential dataset.

    This function reads dataset stored in the folder and returns three lists
    :param data_dir: the path to the potential dataset
    :return:
    files -- the list of file names
    np.array(X) -- the list of potential matrices (in the same order as in files)
    np.array(Y) -- the list of target value (in the same order as in files)
    """
    files, X, Y = [], [], []
    for file in os.listdir(data_dir):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    rf = Pipeline([('vectorizer', MyPotentialTransformer()), ("regressor", ExtraTreesRegressor(n_estimators=200,
                                                                                               criterion="absolute_error",
                                                                                               max_features="log2"))])
    rf.fit(X_train, Y_train)
    predictions = rf.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
