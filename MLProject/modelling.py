import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    roc_auc_score, log_loss, f1_score
)
import mlflow
import mlflow.sklearn
import dagshub
from dotenv import load_dotenv


def modelling_with_tuning(data_path):
    # load & split
    df = pd.read_csv(data_path)
    X = df.drop("HeartDisease", axis=1)
    y = df["HeartDisease"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
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
    return search, search.best_params_, metrics, (y_test, y_pred, y_proba)

if __name__ == "__main__":
    # konfigurasu dagshub
    #load_dotenv()
    dagshub_username = os.getenv('DAGSHUB_USERNAME')
    dagshub_token = os.getenv('DAGSHUB_TOKEN')
    
    if not dagshub_username or not dagshub_token:
        raise ValueError("DAGSHUB_USERNAME atau DAGSHUB_TOKEN tidak terdeteksi!")
    
    dagshub.init(
        repo_owner='dk1781',
        repo_name='heart_attack_mlflow',
        mlflow=True,
        username=dagshub_username,
        password=dagshub_token
    )
    mlflow.set_experiment("HeartAttack_tuning")

    with mlflow.start_run(run_name="Modelling_tuning_manuallog"):
        model, best_params, metrics, preds = modelling_with_tuning("heart_preprocessed.csv")
        # log params & metrics
        for k,v in best_params.items():
            mlflow.log_param(k, v)
        for k,v in metrics.items():
            mlflow.log_metric(k, v)

        # log model
        mlflow.sklearn.log_model(model, "randomforest_bestmodel")

        
