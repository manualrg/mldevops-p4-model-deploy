import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris


from starter.ml.model import train_model, inference
from starter.ml.data import process_data

@pytest.fixture(scope="session")
def raw_data():
    data = pd.DataFrame([[1, "A", "yes"], [2, "B", "no" ], [1, "B", "no"], [2, "A", "no"]],
                     columns=["x1", "x2", "y"])
    return data


@pytest.fixture(scope="session")
def train_data():
    X, y = load_iris(return_X_y=True)
    y = (pd.Series(y) == 0).astype(float).values

    return X, y

@pytest.fixture(scope="session")
def trained_model(train_data):
    X, y = train_data
    clf = train_model(X, y)
    return clf


def test_process_data(raw_data):
    X, y, encoder, lb = process_data(X=raw_data, categorical_features=["x2"], label="y", training=True)

    assert isinstance(X, np.ndarray), f"X is not instance of np.ndarray. found: {type(X)}"
    assert isinstance(y, np.ndarray),  f"y is not instance of np.ndarray. found: {type(y)}"


def test_train_model(train_data, trained_model):
    X, y = train_data
    score = trained_model.score(X, y)

    assert score >= y.mean()


def test_inference(train_data, trained_model):
    X, _ = train_data
    preds = inference(trained_model, X)

    assert isinstance(preds, np.ndarray)