import numpy as np


class Preprocessor:
    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype
        self.column_sizes = None

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        self.column_sizes = X.nunique().values

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        return np.hstack([np.eye(self.column_sizes[i])[np.unique(X[column].values, return_inverse=True)[1]]
                          for i, column in enumerate(X)])

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.stats_table = None

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """

        self.stats_table = [{} for _ in range(X.shape[1])]

        for i, column in enumerate(X):
            for element in np.unique(X[column]):
                statistics = np.zeros(shape=(3, ))

                statistics[0] = np.mean(Y[X[column] == element])
                statistics[1] = Y[X[column] == element].count() / Y.size

                self.stats_table[i][element] = statistics

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """

        res = np.zeros(shape=(X.shape[0], 3 * X.shape[1]))

        for i, row in X.iterrows():
            row_statistics = []
            for j, column in enumerate(X):
                statistics = self.stats_table[j][row[column]]

                statistics[2] = (statistics[0] + a) / (statistics[1] + b)

                row_statistics.extend(statistics)

            res[i] = row_statistics

        return res

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.stats_table = None
        self.folds = None

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """

        self.folds = []
        self.stats_table = [{} for _ in range(self.n_folds)]

        folds = group_k_fold(X.shape[0], self.n_folds, seed)

        for fold_idx, fold in enumerate(folds):
            self.folds.append(fold[0])

            X_fold = X.iloc[fold[1]]
            Y_fold = Y.iloc[fold[1]]

            for column in X:
                self.stats_table[fold_idx][column] = {}

                for _, row in X_fold.iterrows():
                    statistics = np.zeros(shape=(3,))

                    element = row[column]

                    statistics[0] = Y_fold[X_fold[column] == element].mean()
                    statistics[1] = Y_fold[X_fold[column] == element].count() / Y_fold.size

                    self.stats_table[fold_idx][column][element] = statistics

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3]
        """

        res = np.zeros(shape=(X.shape[0], X.shape[1], 3))

        for fold_idx, fold in enumerate(self.folds):
            X_fold = X.iloc[fold]

            for column_idx, column in enumerate(X_fold):
                for row_idx, row in X_fold.iterrows():
                    statistics = self.stats_table[fold_idx][column][row[column]]

                    statistics[2] = (statistics[0] + a) / (statistics[1] + b)

                    res[row_idx][column_idx] = statistics

        res = res.reshape((res.shape[0], -1))

        return res[X.index]

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """

    w = np.zeros(np.unique(x).shape[0])

    for i, element in enumerate(np.unique(x)):
        w[i] = np.mean(y[x == element])

    return w
