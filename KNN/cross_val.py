import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    ids = np.arange(num_objects)
    size = num_objects // num_folds
    split_list = [(np.concatenate((ids[:i * size], ids[(i + 1) * size:])), ids[i * size:(i + 1) * size]) for i in range(num_folds - 1)]
    split_list.append((ids[:(num_folds - 1) * size], ids[(num_folds - 1) * size:]))
    return split_list


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    scores = dict()
    score = 0
    for n in parameters['n_neighbors']:
        for metric in parameters['metrics']:
            for weights in parameters['weights']:
                for normalizer in parameters['normalizers']:
                    for learn, test in folds:
                        model = knn_class(n_neighbors=n, weights=weights, metric=metric)
                        x_learn, x_test = X[learn], X[test]
                        if normalizer[0]:
                            normalizer[0].fit(x_learn)
                            x_learn = normalizer[0].transform(x_learn)
                            x_test = normalizer[0].transform(x_test)
                        y_learn, y_test = y[learn], y[test]
                        model.fit(x_learn, y_learn)
                        y_predict = model.predict(x_test)
                        score += score_function(y_test, y_predict)
                    score /= len(folds)
                    scores[normalizer[1], n, metric, weights] = score
                    score = 0

    return scores
