import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """

    X = train_features[:, 3:5]
    y = train_target

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    clf = SVC(kernel="rbf", C=0.5)

    clf.fit(X, y)

    test_features = test_features[:, 3:5]
    test_features = scaler.transform(test_features)

    return clf.predict(test_features)
