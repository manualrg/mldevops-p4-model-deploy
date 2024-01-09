# Script to train machine learning model.
from pathlib import Path
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.data import read_data, process_data
from ml.model import train_model, inference, compute_model_metrics, compute_model_metrics_on_slices

import os
print(os.getcwd())
import  utils

def main():


    # Add code to load in the data.

    df_raw = read_data(path=utils.path_data, file_name="census.csv")

    df_dev = df_raw
        

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    df_train, df_test = train_test_split(df_dev, test_size=0.20)



    X_train, y_train, encoder, lb = process_data(
        df_train, categorical_features=utils.cat_features, label=utils.target, training=True
    )

    # Proces the test data with the process_data function.
    X_test, y_test, encoder, lb = process_data(
        df_test, categorical_features=utils.cat_features, label=utils.target, training=False, encoder=encoder, lb=lb
    )


    # Train and save a model.
    clf = train_model(X_train, y_train)


    preds = inference(clf, X_test)

    precision, recall, fbeta = compute_model_metrics(y_test, preds)

    print(f"Model f1-score on test data: {fbeta}")

    df_metrics_sex_slice = compute_model_metrics_on_slices(df_test, y_test, preds, feature="sex")

    fln_model = utils.path_model /  "model.pickle"
    fln_cv_summary = utils.path_model /  "cv_summary.csv"
    fln_encoder = utils.path_model /  "encoder.pickle"
    fln_label_encoder = utils.path_model /  "label_encoder.pickle"

    fln_metrics_slices = utils.path_model /  "metrics_by_slice.csv"

    with open(fln_model, "wb") as file:
        pickle.dump(clf.best_estimator_, file)
    with open(fln_encoder, "wb") as file:
        pickle.dump(encoder, file)
    with open(fln_label_encoder, "wb") as file:
        pickle.dump(lb, file)

    df_metrics_sex_slice.to_csv(fln_metrics_slices, index=False)
    pd.DataFrame(clf.cv_results_).to_csv(fln_cv_summary, index=False)

if __name__ == "__main__":
    main()