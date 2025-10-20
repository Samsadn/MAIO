# training entrypoint (v0.1 / v0.2)
import os
import json
import joblib

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

from .config import CONFIG


def _resolve_version() -> str:
    """
    Resolve the model version:
    - Prefer MODEL_VERSION from environment (release workflow),
    - Otherwise use CONFIG.model_version.
    """
    ver = os.getenv("MODEL_VERSION", CONFIG.model_version)
    return str(ver)


def _build_pipeline(version: str) -> tuple[Pipeline, str]:
    """
    Return a (pipeline, model_type_name) tuple based on the resolved version.
      - v0.1*: StandardScaler + LinearRegression
      - v0.2*: StandardScaler + Ridge(alpha=1.0)
    Accepts versions with or without 'v' prefix, e.g., "v0.2.0" or "0.2.0".
    """
    v = version.lower().lstrip("v")
    if v.startswith("0.2"):
        model = Ridge(alpha=1.0, random_state=CONFIG.seed)
        model_type = "Ridge"
    else:
        model = LinearRegression()
        model_type = "LinearRegression"

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", model),
    ])
    return pipe, model_type


def train_and_save_model():
    """
    Load data, train the pipeline for the resolved version, evaluate RMSE,
    and save artifacts (model + metrics) to CONFIG paths.
    """
    # Ensure artifacts directory exists
    CONFIG.artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Resolve version & build pipeline
    version = _resolve_version()
    pipeline, model_type = _build_pipeline(version)

    # 1) Load data
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    # 2) Split (reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=CONFIG.seed
    )

    # 3) Train
    print(f"[train] Starting training for version={version} ({model_type})")
    pipeline.fit(X_train, y_train)
    print("[train] Training complete.")

    # 4) Evaluate (RMSE; version-agnostic: MSE -> sqrt)
    y_pred = pipeline.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(mse ** 0.5)

    # 5) Save artifacts
    joblib.dump(pipeline, CONFIG.model_path)
    print(f"[train] Model saved to {CONFIG.model_path}")

    metrics = {
        "version": version if str(version).startswith("v") else f"v{version}",
        "rmse": rmse,
        "random_state": CONFIG.seed,
        "model_type": model_type,
    }
    CONFIG.metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[train] Metrics saved to {CONFIG.metrics_path}")
    print(f"[train] RMSE: {rmse:.4f}")

    return metrics


def main():
    """Entrypoint used by tests and CI/CD."""
    return train_and_save_model()


if __name__ == "__main__":
    main()
