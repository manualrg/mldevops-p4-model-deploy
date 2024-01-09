from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import pandas as pd 

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = LogisticRegression(penalty="l2", random_state=123, fit_intercept=True, max_iter=5000)
    param_grid = {"C": [1e-1, 1, 10, 100]}
    clf_gs = GridSearchCV(model, param_grid, scoring='f1', cv=5, n_jobs=5)
    clf_gs.fit(X_train, y_train)


    return clf_gs


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_metrics_on_slices(Xs, y, preds, feature):
    assert len(Xs) == len(y)
    assert len(y) == len(preds)
    assert feature in Xs.columns

    X_slices =  Xs[[feature]].copy()
    X_slices["target"] = y
    X_slices["preds"] = preds

    slices = X_slices[feature].value_counts(dropna=False)
    stats = slices.to_frame("n_rows")
    stats.index.name = "levels"

    for idx, _ in slices.iteritems():
        Xs_slice = X_slices.loc[X_slices[feature] == idx]
        metrics = compute_model_metrics(y=Xs_slice["target"], preds=Xs_slice["preds"] )
        stats.loc[idx, ["precision", "recall", "f1"]] = metrics
    
    stats["feature"] = feature

    return stats.reset_index()[["feature", "levels", "n_rows", "precision", "recall", "f1"]]


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
