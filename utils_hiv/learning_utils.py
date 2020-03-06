import pickle
import json
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, recall_score, accuracy_score, precision_score, balanced_accuracy_score
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.dummy import DummyClassifier, DummyRegressor
from scipy.stats import pearsonr

from .data_utils import split_data, subsampling_balance
from .metrics import *

class NoCrashModel:
    def __init__(self, *args, **kwargs):
        pass
    def fit(self, *args, **kwargs):
        pass
    def predict(self, X, *args, **kwargs):
        return ['This model does not exist'] * len(X)

REGRESSION_TARGETS = [
    'ABC', 'AZT', 
    'FTC', '3TC', 
    'TDF', 'DOR', 
    'EFV', 'ETR', 
    'NVP', 'RPV'
]

CLASSIFICATION_MODELS = {
    'RF': RandomForestClassifier,
    'Logistic': LogisticRegressionCV,
    'KNN': KNeighborsClassifier,
    'Bayes': MultinomialNB,
    'Complement': ComplementNB
}

REGRESSION_MODELS = {
    'RF': RandomForestRegressor,
    'Lasso': LassoCV
}

def get_sets(split_path, target):
    train, test = pickle.load(open(split_path, 'rb'))
    return split_data(train, target), split_data(test, target)


def pair_measure_with_features(features, measures):
    paired = {}
    for feature, measure in zip(features, measures):
        paired[feature] = measure
    return paired


def train_model(model_type, train_set, params_path, target, balance=False):

    if target in REGRESSION_TARGETS:
        target = f"{target}_Score"
        models = REGRESSION_MODELS
    else:
        models = CLASSIFICATION_MODELS

    if model_type in ['Bayes', 'Complement']:
        params = dict()
    elif isinstance(params_path, dict):
        params = params_path
    else:
        params = json.load(open(params_path, 'r'))
    data = pd.read_csv(train_set, sep='\t', header=0, index_col=0)

    if balance:
        data = subsampling_balance(data, target)

    X_train, y_train = split_data(data, target)

    model = models.get(model_type, NoCrashModel)(**params)
    model.fit(X_train, y_train)

    coefs = get_coefficients(model, X_train.columns.tolist())

    return model, coefs

def get_predictions(model, test_set, target, balance=False):
    data = pd.read_csv(test_set, sep='\t', header=0, index_col=0)
    
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
        df['pred'] = predictions
        df['real'] = y_test
    else:
        df = pd.DataFrame(predictions, columns=['pred'], index=X_test.index)
        df['real'] = y_test

    return df

def get_coefficients(model, features):
    index = model.classes_ if len(model.classes_) > 2 else model.classes_[-1:]
    if isinstance(model, RandomForestClassifier) or isinstance(model, RandomForestRegressor):
        coefs = [model.feature_importances_]*len(index)
    else:
        coefs = model.coef_
    return pd.DataFrame(coefs, index=index, columns=features)
