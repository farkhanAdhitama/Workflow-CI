import pandas as pd
import mlflow
import mlflow.sklearn
import sys
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

mlflow.set_tracking_uri("http://127.0.0.1:8000")
mlflow.sklearn.autolog(disable=True)

# Get parameters from command line
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
test_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

# Load dataset
df = pd.read_csv("MLProject/bank_customer_preprocessed.csv")
df = df.dropna()

# Split features and target
X = df.drop("Exited", axis=1)
y = df["Exited"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

# MLflow experiment
mlflow.set_experiment("Customer_Churn_Pred")

with mlflow.start_run():
    # Inisialisasi model
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Prediksi - hanya untuk training
    y_train_pred = model.predict(X_train)
    y_train_proba = model.predict_proba(X_train)

    # Calculate metrics - hanya training
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    train_precision = precision_score(y_train, y_train_pred)
    train_recall = recall_score(y_train, y_train_pred)
    train_roc_auc = roc_auc_score(y_train, y_train_proba[:, 1])
    train_log_loss_val = log_loss(y_train, y_train_proba)

    # Manual logging parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
    mlflow.log_param("random_state", 42)

    # Manual logging metrics - Training saja
    mlflow.log_metric("training_accuracy_score", train_accuracy)
    mlflow.log_metric("training_f1_score", train_f1)
    mlflow.log_metric("training_log_loss", train_log_loss_val)
    mlflow.log_metric("training_precision_score", train_precision)
    mlflow.log_metric("training_recall_score", train_recall)
    mlflow.log_metric("training_roc_auc", train_roc_auc)
    mlflow.log_metric("training_score", model.score(X_train, y_train))

    # Simpan model ke MLflow
    mlflow.sklearn.log_model(
        model, "model", registered_model_name="RandomForest_CustomerChurn"
    )

    # Save model locally untuk artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f"model_{timestamp}.joblib"
    joblib.dump(model, model_file)

    # Save metadata
    metadata = {
        "model_info": {
            "accuracy": train_accuracy,
            "timestamp": timestamp,
            "model_file": model_file,
        },
        "parameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_size": test_size,
            "random_state": 42,
        },
        "training_metrics": {
            "accuracy": train_accuracy,
            "f1_score": train_f1,
            "precision": train_precision,
            "recall": train_recall,
            "roc_auc": train_roc_auc,
            "log_loss": train_log_loss_val,
        },
        "dataset_info": {
            "shape": f"{df.shape[0]}x{df.shape[1]}",
            "features": list(X.columns),
            "target": "Exited",
        },
    }

    metadata_file = f"metadata_{timestamp}.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
