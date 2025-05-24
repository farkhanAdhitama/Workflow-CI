import pandas as pd
import mlflow
import mlflow.sklearn
import sys
import joblib
import json
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Enable autolog - otomatis log semua parameter, metrics, dan model
mlflow.sklearn.autolog()

# Get parameters from command line
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 10
test_size = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2

# Load dataset
df = pd.read_csv("bank_customer_preprocessed.csv")
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
    # Train model 
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Predict untuk accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log additional custom parameters (opsional)
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")

    # Save model locally untuk artifacts
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f"model_{timestamp}.joblib"
    joblib.dump(model, model_file)

    # Save metadata
    metadata = {
        "accuracy": accuracy,
        "timestamp": timestamp,
        "parameters": {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "test_size": test_size,
        },
    }

    with open(f"metadata_{timestamp}.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model trained with accuracy: {accuracy:.4f}")
    print(f"Model saved as: {model_file}")
