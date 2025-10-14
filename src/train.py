# training entrypoint (v0.1 / v0.2)
import pandas as pd
import json
import joblib

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# --- Configuration for Reproducibility ---
RANDOM_STATE = 42
MODEL_VERSION = "v0.1"
MODEL_PATH = "artifacts/model.joblib"
METRICS_PATH = "artifacts/metrics.json"


def train_and_save_model():
    """
    Loads the diabetes dataset, trains the v0.1 pipeline (StandardScaler + LinearRegression),
    evaluates RMSE, and saves the model and metrics artifacts.
    """
    # 1. Load Data
    # Xy.frame: DataFrame containing features (X) and target (y)
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]
    
    # 2. Split Data (using fixed random state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # 3. Define the v0.1 Baseline Pipeline
    # Constraint: simple StandardScaler + LinearRegression
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # 4. Train Model
    print("Starting model training...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate Performance
    y_pred = pipeline.predict(X_test)
    
    # Calculate Mean Squared Error (MSE) 
    mse = mean_squared_error(y_test, y_pred) 
    # Calculate Root Mean Squared Error (RMSE) by taking the square root 
    rmse = mse ** 0.5
    
    # 6. Save Artifacts
    
    # Save the trained pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save metrics for logging and CHANGELOG.md
    metrics = {
        "version": MODEL_VERSION,
        "rmse": float(rmse),
        "random_state": RANDOM_STATE,
        "model_type": "LinearRegression"
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    train_and_save_model()