import mlflow
import pandas as pd
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import os
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 37

    file_path = (
        sys.argv[3]
        if len(sys.argv) > 3
        else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "diabetes_preprocessing.csv"
        )
    )
    data = pd.read_csv(file_path)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Outcome", axis=1),
        data["Outcome"],
        random_state=42,
        test_size=0.2,
    )

    input_example = X_train[0:5]

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        model.fit(X_train, y_train)

        # Prediksi untuk evaluasi training
        y_train_pred = model.predict(X_train)
        y_train_prob = model.predict_proba(X_train)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=model, artifact_path="model", input_example=input_example
        )

        # Hitung dan log metrik evaluasi
        metrics = {
            "RandomForestClassifier_score_X_test": model.score(X_test, y_test),
            "training_score": model.score(X_train, y_train),
            "training_accuracy_score": accuracy_score(y_train, y_train_pred),
            "training_f1_score": f1_score(y_train, y_train_pred),
            "training_log_loss": log_loss(y_train, y_train_prob),
            "training_precision_score": precision_score(y_train, y_train_pred),
            "training_recall_score": recall_score(y_train, y_train_pred),
            "training_roc_auc": roc_auc_score(y_train, y_train_prob[:, 1]),
        }

        for key, value in metrics.items():
            mlflow.log_metric(key, value)
