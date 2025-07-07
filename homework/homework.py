
# flake8: noqa: E501

import os
import gzip
import json
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

def load_data(file_path):
    return pd.read_csv(file_path, compression="zip")

def clean_data(df):
    df = df.copy()
    df.rename(columns={"default payment next month": "default"}, inplace=True)
    df.drop(columns="ID", inplace=True)
    df = df[(df["EDUCATION"] != 0) & (df["MARRIAGE"] != 0)]
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)
    return df

def split_features_labels(train_df, test_df):
    X_train = train_df.drop(columns="default")
    y_train = train_df["default"]
    X_test = test_df.drop(columns="default")
    y_test = test_df["default"]
    return X_train, y_train, X_test, y_test

def build_pipeline():
    categorical = ["SEX", "EDUCATION", "MARRIAGE"]
    numerical = [
        "LIMIT_BAL", "AGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4",
        "PAY_5", "PAY_6", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
        "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
        "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"
    ]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", StandardScaler(), numerical),
    ])

    return Pipeline([
        ("preprocessor", preprocessor),
        ("feature_selection", SelectKBest(score_func=f_classif)),
        ("pca", PCA()),
        ("classifier", MLPClassifier(max_iter=15000, random_state=17)),
    ])

def tune_hyperparameters(pipeline):
    param_grid = {
        'pca__n_components': [None],
        'feature_selection__k': [20],
        'classifier__hidden_layer_sizes': [(50, 30, 40, 60)],
        'classifier__alpha': [0.26],
        'classifier__learning_rate_init': [0.001],
    }

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=2
    )

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)

def evaluate_model(model, X, y, dataset_label):
    y_pred = model.predict(X)
    metrics = {
        "type": "metrics",
        "dataset": dataset_label,
        "precision": round(precision_score(y, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y, y_pred), 4),
        "recall": round(recall_score(y, y_pred), 4),
        "f1_score": round(f1_score(y, y_pred), 4)
    }
    return metrics, y_pred

def get_confusion_dict(y_true, y_pred, dataset_label):
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_label,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])}
    }

def main():
    os.makedirs("files/output", exist_ok=True)

    df_train = clean_data(load_data("files/input/train_data.csv.zip"))
    df_test = clean_data(load_data("files/input/test_data.csv.zip"))

    X_train, y_train, X_test, y_test = split_features_labels(df_train, df_test)

    pipeline = build_pipeline()
    grid_search = tune_hyperparameters(pipeline)
    model = grid_search.fit(X_train, y_train)

    metrics_train, y_pred_train = evaluate_model(model, X_train, y_train, "train")
    metrics_test, y_pred_test = evaluate_model(model, X_test, y_test, "test")

    cm_train = get_confusion_dict(y_train, y_pred_train, "train")
    cm_test = get_confusion_dict(y_test, y_pred_test, "test")

    with open("files/output/metrics.json", "w") as f:
        for entry in [metrics_train, metrics_test, cm_train, cm_test]:
            f.write(json.dumps(entry) + "\n")

    save_model(model, "files/models/model.pkl.gz")

if __name__ == "__main__":
    main()
