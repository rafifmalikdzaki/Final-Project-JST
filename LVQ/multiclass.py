import numpy as np
from sklearn.preprocessing import LabelBinarizer
from copy import *


def _fit_binary(estimator, X, y, classes=None):
    estimator = deepcopy(estimator)
    estimator.fit(X, y)
    return estimator


class OneVsRestClassifier:
    def __init__(self, estimator):
        self.classes_ = None
        self.estimator = estimator
        self.estimators_ = []
        self.label_bin_ = None

    def fit(self, X, y):
        self.label_bin_ = LabelBinarizer(sparse_output=True)
        Y = self.label_bin_.fit_transform(y)
        Y = Y.tocsc()
        self.classes_ = self.label_bin_.classes_
        columns = (col.toarray().ravel() for col in Y.T)
        for i, column in enumerate(columns):
            self.estimators_.append(
                _fit_binary(self.estimator, X, column)
            )

    def predict(self, X):
        n_samples = X.shape[0]

        maxima = np.empty(n_samples, dtype=np.double)
        maxima.fill(-np.inf)
        argmaxima = np.zeros(n_samples, dtype=np.int8)
        for i, e in enumerate(self.estimators_):
            pred = e.predict(X)
            np.maximum(maxima, pred, out=maxima)
            argmaxima[maxima == pred] = i
        return self.classes_[argmaxima]


class OneVsOneClassifier:
    pass


class SoftMax:
    pass


class NearestClassMean:
    pass
