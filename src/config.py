import pathlib
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    # Reproducibility
    seed: int = 42
    # Model versioning (bump for v0.2)
    model_version: str = "0.2.0"
    # Artifacts
    artifacts_dir: pathlib.Path = pathlib.Path("artifacts")
    model_path: pathlib.Path = artifacts_dir / "model.joblib"
    metrics_path: pathlib.Path = artifacts_dir / "metrics.json"

CONFIG = AppConfig()
