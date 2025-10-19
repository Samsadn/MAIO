# training entrypoint (v0.1 / v0.2)
import json
import joblib
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Import shared configuration
from .config import CONFIG


def train_and_save_model():
    """
    Loads the diabetes dataset, trains the v0.1 pipeline (StandardScaler + LinearRegression),
    evaluates RMSE, and saves the model and metrics artifacts defined in CONFIG.
    """
    # Ensure artifacts directory exists
    CONFIG.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    # 2. Split data (using CONFIG.seed for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG.seed
    )

    # 3. Define baseline pipeline (v0.1)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', LinearRegression())
    ])

    # 4. Train model
    print("Starting model training...")
    pipeline.fit(X_train, y_train)
    print("Training complete.")

    # 5. Evaluate performance
    y_pred = pipeline.predict(X_test)
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))

    # 6. Save artifacts
    joblib.dump(pipeline, CONFIG.model_path)
    print(f"Model saved to {CONFIG.model_path}")

    metrics = {
        "version": CONFIG.model_version,
        "rmse": rmse,
        "random_state": CONFIG.seed,
        "model_type": "LinearRegression"
    }
    CONFIG.metrics_path.write_text(json.dumps(metrics, indent=4))
    print(f"Metrics saved to {CONFIG.metrics_path}")

    return metrics


def main():
    """Wrapper entrypoint so tests and CI can call train.main()."""
    return train_and_save_model()


if __name__ == "__main__":
    main()
