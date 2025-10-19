# Changelog

All notable changes to this project will be documented in this file.

---

## [v0.2.1] – 2025-10-19
### Changed
- Fixed lowercase image tag issue for GHCR push in `release.yml`.
- Added `httpx` dependency to enable FastAPI `TestClient` in CI tests.
- Replaced `mean_squared_error(..., squared=False)` with manual RMSE computation to ensure compatibility.
- Removed unused imports and cleaned up code style to satisfy `ruff` linter.
- Integrated `config.py` into `train.py` for unified configuration (seed, versioning, artifact paths).
- Improved workflow stability and reproducibility for CI/CD and release pipelines.

---

## [v0.2.0] – 2025-10-14
### Added
- Introduced `Dockerfile` for containerising the ML service.
- Implemented smoke tests for `/health` and `/predict` endpoints during Docker image build.
- Added automatic model retraining in the release workflow before publishing to GHCR.

### Changed
- Refined GitHub Actions structure with separate `ci.yml` and `release.yml`.
- Updated pipeline to include artifact checks (`model.joblib`, `metrics.json`).

---

## [v0.1.0] – 2025-10-10
### Added
- Initial project setup:
  - FastAPI application with `/health` and `/predict` endpoints.
  - Scikit-learn training pipeline (`StandardScaler` + `LinearRegression`).
  - Unit tests and smoke tests.
  - CI workflow (`ci.yml`) for linting, testing, and artifact storage.
  - Documentation files: `README.md`, `CHANGELOG.md`, and `requirements.txt`.

---