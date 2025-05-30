import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, log_loss, f1_score
)
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv


def modelling_with_tuning(X_train_path, X_test_path, y_train_path, y_test_path):
    
    # Load data
    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).squeeze()
    y_test = pd.read_csv(y_test_path).squeeze()
    # hyperparameter tuning
    param_dist = {
        "n_estimators": [50, 100, 200, 400],
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.8],

    }
    rf = RandomForestClassifier(random_state=42)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=10, cv=3, scoring="accuracy", random_state=42
    )
    search.fit(X_train, y_train)

    # predictions & probabilities
    y_pred = search.predict(X_test)
    y_proba = search.predict_proba(X_test)[:,1]

    # compute metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1score" : f1_score(y_test,y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba)
    }
    return search, search.best_params_, metrics, X_test

if __name__ == "__main__":
    X_train_path = "heart_preprocessing/X_train.csv"
    X_test_path = "heart_preprocessing/X_test.csv"
    y_train_path = "heart_preprocessing/y_train.csv"
    y_test_path = "heart_preprocessing/y_test.csv"
    load_dotenv()
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if not username or not password:
        raise EnvironmentError("MLFLOW_TRACKING_USERNAME dan MLFLOW_TRACKING_PASSWORD harus di-set sebagai environment variable")

    mlflow.set_tracking_uri("https://dagshub.com/dk1781/heart_attack_mlflow.mlflow/")
    mlflow.set_experiment("HeartAttack_tuning")

    with mlflow.start_run(run_name="Modelling_tuning_manuallog1"):
        model, best_params, metrics, X_test = modelling_with_tuning(X_train_path, X_test_path, y_train_path, y_test_path)
        # log params & metrics
        for k,v in best_params.items():
            mlflow.log_param(k, v)
        for k,v in metrics.items():
            mlflow.log_metric(k, v)

        # log model
        input_example = X_test.iloc[:1]
        signature = infer_signature(X_test, model.predict(X_test))

        # Simpan model ke dalam MLflow dengan nama artifact rf_model
        mlflow.sklearn.log_model(
            model,
            artifact_path="randomforest_bestmodel",
            signature=signature,
            input_example=input_example
        )

