import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder


def chi2_selection(X, y, k=10):
    """
    Select features according to the k highest scores.
    :param X: {array-like, sparse matrix}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and
        n_features is the number of features.
    :param y: array-like, shape = [n_samples]
        Target values.
    :param k: int or "all", optional (default=10)
        Number of top features to select. The "all" option bypasses selection, for use in a parameter search.
    :return: array-like, shape = [n_samples, n_features]
        The selected features.
    """
    X_new = SelectKBest(chi2, k=k).fit_transform(X, y)
    return X_new


def min_max_scaler(X):
    """
    Transform features by scaling each feature to a given range.
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The data to scale, element by element.
    :return: array-like, shape (n_samples, n_features)
        The scaled features.
    """
    X_new = MinMaxScaler().fit_transform(X)
    return X_new


def standard_scaler(X):
    """
    Standardize features by removing the mean and scaling to unit variance.
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The data to center and scale.
    :return: array-like, shape (n_samples, n_features)
        The scaled features.
    """
    X_new = StandardScaler().fit_transform(X)
    return X_new


def robust_scaler(X):
    """
    Scale features using statistics that are robust to outliers.
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The data to center and scale.
    :return: array-like, shape (n_samples, n_features)
        The scaled features.
    """
    X_new = RobustScaler().fit_transform(X)
    return X_new


def max_abs_scaler(X):
    """
    Scale each feature by its maximum absolute value.
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The data to center and scale.
    :return: array-like, shape (n_samples, n_features)
        The scaled features.
    """
    X_new = MaxAbsScaler().fit_transform(X)
    return X_new


def quantile_transformer(X):
    """
    Transform features using quantiles information.
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The data to transform.
    :return: array-like, shape (n_samples, n_features)
        The transformed features.
    """
    X_new = QuantileTransformer().fit_transform(X)
    return X_new


def power_transformer(X):
    """
    Apply a power transform featurewise to make data more Gaussian-like.
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The data to transform.
    :return: array-like, shape (n_samples, n_features)
        The transformed features.
    """
    X_new = PowerTransformer().fit_transform(X)
    return X_new


def normalizer(X):
    """
    Normalize samples individually to unit norm.
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The data to normalize, row-wise.
    :return: array-like, shape (n_samples, n_features)
        The normalized features.
    """
    X_new = Normalizer().fit_transform(X)
    return X_new


def binarizer(X):
    """
    Binarize data (set feature values to 0 or 1) according to a threshold.
    :param X: {array-like, sparse matrix}, shape (n_samples, n_features)
        The data to binarize, element by element.
    :return: array-like, shape (n_samples, n_features)
        The binarized features.
    """
    X_new = Binarizer().fit_transform(X)
    return X_new
for i in range(1, 11):

    X_new = chi2_selection(X, y, k=i)
    print(X_new.shape)
    for j in range(0, i):
        print(X_new[:, j])
        continue
    continue

X_new = min_max_scaler(X)
print(X_new.shape)
for i in range(0, 10):
        X = pd.get_dummies(df2[i])
        print(X.shape)
        continue
        # X=X.fillna(0)
        # y=y.fillna(0)
        # X_new = chi2_selection(X, y, k=10)
        # print(X_new.shape)
        # for j in range(0, 10):
        #     print(X_new[:, j])
        #     continue
        # continue

        # X_new = min_max_scaler(X)
        # print(X_new.shape)
        # for j in range(0, 10):
        #     print(X_new[:, j])
        #     continue
        # continue


        scaler = MinMaxScaler()
        X_new = scaler.fit_transform(X)
        print(X_new.shape)
        str_score = str(np.sum(X_new))
        len_x = len(X_new)
        len_y = len(X_new[0])
        str_len_x = str(len_x)
        str_len_y = str(len_y)
        crit = stats.chi2.ppf(q=0.99, df=(len_x - 1) * (len_y - 1))
        str_write = i + ' & ' + j + ' chi2 is: ' + str_score + '  lenx: ' + str_len_x + '  leny: ' + str_len_y + ' crit0.99: ' + str(
            crit) + '\n'
        # print(str_write)
        f2.write(str_write)
        print(i, '&', j, 'chi2 is:', np.sum(sk.scores_), '  lenx:', str_len_x, 'leny:', str_len_y, '  crit0.99:', crit)
