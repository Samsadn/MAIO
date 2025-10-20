# Changelog

All notable changes to this project are documented in this file.  
Each release corresponds to a tagged version built, tested, and deployed via GitHub Actions and GHCR.

---

## [v0.2.0] – 2025-10-18
### 🎯 Iteration 2 – Model Improvement & CI Polish

#### 🚀 Changes
- Upgraded model from **LinearRegression** → **Ridge(alpha=1.0)** for improved generalization.
- Integrated model versioning via `MODEL_VERSION` environment variable.
- Enhanced `/health` endpoint to report the actual trained model version.
- Improved training logs and artifact consistency.
- Refined CI/CD pipelines for reproducibility and cleaner metrics outputs.

#### 📊 Model Performance Comparison
| Metric | v0.1.0 (LinearRegression) | v0.2.0 (Ridge) | Δ Improvement |
|---------|----------------------------|----------------|----------------|
| **RMSE** | 54.82 | **53.78** | ↓ 1.9 % |
| **Seed** | 42 | 42 | — |
| **Data Split** | 80/20 fixed | 80/20 fixed | — |
| **Model Type** | `LinearRegression` | `Ridge(alpha=1.0)` | ✅ Regularized |

> The Ridge model slightly reduced RMSE and is expected to generalize better on unseen patient data due to regularization.

#### 🧱 Technical Notes
- Rebuilt image: `ghcr.io/samsadn/maio:v0.2.0`
- Artifacts: `artifacts/model.joblib` and `artifacts/metrics.json` (baked in release image)
- CI/CD smoke tests passed (`/health` = 200, `/predict` = 200)

---

## [v0.1.0] – 2025-10-10
### 🎬 Initial Baseline Release
- Implemented **StandardScaler + LinearRegression** baseline model.
- Established GitHub Actions CI pipeline:
  - Linting (`ruff`)
  - Unit tests (`pytest`)
  - Artifact upload (`model.joblib`, `metrics.json`)
- Built and published Docker image `ghcr.io/samsadn/maio:v0.1.0`.
- Added FastAPI endpoints:
  - `/health` – returns service status and model version
  - `/predict` – returns numeric disease progression prediction
- Initial RMSE: **54.82**

---

### 🧭 Versioning Policy
Releases follow **semantic versioning**:
- `MAJOR.MINOR.PATCH`
- Each tagged version automatically triggers retraining, Docker build, and GHCR publish via `release.yml`.

### 📦 Repository Artifacts
All artifacts are stored under `/artifacts`:
- `model.joblib` – trained model pipeline  
- `metrics.json` – evaluation metrics  
- Logged and attached in each GitHub Release.

---
