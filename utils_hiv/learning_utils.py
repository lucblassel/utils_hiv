import json
import os
import pickle

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

from .data_utils import split_data, subsampling_balance
from .metrics import *

HERE = os.path.dirname(__file__)


class NoCrashModel:
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X, *args, **kwargs):
        return ["This model does not exist"] * len(X)


class FisherTestModel:
    CORRECTIONS = ["Bonferroni", "fdr_bh", "fdr_by"]

    def __init__(self, subtype, correction, n_vote, alpha=0.05):
        if correction not in FisherTestModel.CORRECTIONS:
            raise ValueError(
                f"correction must be one of following: {FisherTestModel.CORRECTIONS}"
            )
        self.n_vote = n_vote
        self.correction, self.subtype, self.alpha = correction, subtype, alpha
        self.mutations = self._read_file(correction, subtype, alpha)
        self.classes_ = [0, 1]

    def _read_file(self, correction, subtype, alpha):
        mutations = pd.read_csv(
            os.path.join(HERE, "data/fisher_p_values.tsv"), sep="\t", index_col=0
        )
        subset = mutations[(mutations["subtype"] == subtype)][[correction]]
        return subset

    def fit(self, *args, **kwargs):
        pass

    def predict(self, X, *args, **kwargs):
        sign = self.mutations[
            self.mutations[self.correction] <= self.alpha
        ].index.tolist()
        sub = X.filter(sign, axis=1)
        presence = sub.sum(axis=1)
        return (presence >= self.n_vote).astype(int)


class FisherBonf1(FisherTestModel):
    def __init__(self, subtype, alpha=0.05):
        super().__init__(subtype, "Bonferroni", 1, alpha)


class FisherBonf2(FisherTestModel):
    def __init__(self, subtype, alpha=0.05):
        super().__init__(subtype, "Bonferroni", 2, alpha)


class FisherBH1(FisherTestModel):
    def __init__(self, subtype, alpha=0.05):
        super().__init__(subtype, "fdr_bh", 1, alpha)


class FisherBH2(FisherTestModel):
    def __init__(self, subtype, alpha=0.05):
        super().__init__(subtype, "fdr_bh", 2, alpha)


STAT_MODELS = {
    "FisherBonf1": FisherBonf1,
    "FisherBonf2": FisherBonf2,
    "FisherBH1": FisherBH1,
    "FisherBH2": FisherBH2,
}

REGRESSION_TARGETS = [
    "ABC",
    "AZT",
    "FTC",
    "3TC",
    "TDF",
    "DOR",
    "EFV",
    "ETR",
    "NVP",
    "RPV",
]

CLASSIFICATION_MODELS = {
    "RF": RandomForestClassifier,
    "Logistic": LogisticRegressionCV,
    "KNN": KNeighborsClassifier,
    "Bayes": MultinomialNB,
    "Complement": ComplementNB,
    "FisherBonf1": FisherBonf1,
    "FisherBonf2": FisherBonf2,
    "FisherBH1": FisherBH1,
    "FisherBH2": FisherBH2,
}

REGRESSION_MODELS = {"RF": RandomForestRegressor, "Lasso": LassoCV}


def get_sets(split_path, target):
    train, test = pickle.load(open(split_path, "rb"))
    return split_data(train, target), split_data(test, target)


def pair_measure_with_features(features, measures):
    paired = {}
    for feature, measure in zip(features, measures):
        paired[feature] = measure
    return paired


def train_model(model_type, train_set, params_path, target, subtype, balance=False):

    if target in REGRESSION_TARGETS:
        target = f"{target}_Score"
        models = REGRESSION_MODELS
    else:
        models = CLASSIFICATION_MODELS

    if model_type in ["Bayes", "Complement"]:
        params = dict()
    elif model_type in STAT_MODELS.keys():
        params = {"subtype": subtype}
    elif isinstance(params_path, dict):
        params = params_path
    else:
        params = json.load(open(params_path, "r"))
    data = pd.read_csv(train_set, sep="\t", header=0, index_col=0)

    if balance:
        data = subsampling_balance(data, target)

    X_train, y_train = split_data(data, target)

    model = models.get(model_type, NoCrashModel)(**params)
    model.fit(X_train, y_train)

    coefs = get_coefficients(model, X_train.columns.tolist())

    return model, coefs


def get_predictions(model, test_set, target, balance=False):
    data = pd.read_csv(test_set, sep="\t", header=0, index_col=0)

    if target in REGRESSION_TARGETS:
        target = f"{target}_Score"

    if balance:
        data = subsampling_balance(data, target)

    X_test, y_test = split_data(data, target)

    predictions = model.predict(X_test)

    if getattr(model, "predict_proba", None):
        probabilities = model.predict_proba(X_test)
        df = pd.DataFrame(probabilities, columns=model.classes_)
        df.set_index(X_test.index, inplace=True)
        df["pred"] = predictions
        df["real"] = y_test
    else:
        df = pd.DataFrame(predictions, columns=["pred"], index=X_test.index)
        df["real"] = y_test

    return df


def get_coefficients(model, features):
    index = model.classes_ if len(model.classes_) > 2 else model.classes_[-1:]
    if isinstance(model, RandomForestClassifier) or isinstance(
        model, RandomForestRegressor
    ):
        coefs = [model.feature_importances_] * len(index)
    elif np.array([isinstance(model, clf) for clf in STAT_MODELS.values()]).any():
        coefs = model.mutations.copy()
        coefs["pos"] = 1 - coefs.iloc[:, 0]
        coefs.columns = [0, 1]
        return coefs.transpose()
    else:
        coefs = model.coef_
    return pd.DataFrame(coefs, index=index, columns=features)
