````markdown
# Assignment 3 - MLOps

## Case: Virtual Diabetes Clinic Triage 

### Context
A hospital runs a virtual diabetes clinic. Each week, nurses review hundreds of patient check-ins (vitals, labs, lifestyle notes) to decide who needs a follow-up call. Reviews are manual and time-consuming.

### Goal
Build a small ML service that predicts short-term disease progression and returns a continuous risk score. The clinic will use the score to prioritize follow-ups. Everything should be built in a pipeline using [GitHub Actions](https://docs.github.com/en/actions).

### Data
For the assignment, use the open scikit-learn Diabetes regression dataset as a stand-in for de-identified EHR features:

```python
from sklearn.datasets import load_diabetes

Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]  # acts as a "progression index" (higher = worse)
````

Treat `y` as a progression index: higher values ≈ greater deterioration risk (e.g., rising HbA1c/complications risk).
In production, a clinic would train on real outcomes; here, we map the same mechanics to a safe, open dataset.

---

## Users & Flow

* **Triage Nurse** opens a dashboard sorted by predicted progression (descending).
* **ML Service** hosts `/predict` that the dashboard calls for each patient.
* **MLOps Team (you)** owns training, packaging, testing, and releasing the service.

---

## Non-functional Constraints

* **Portability:** Docker image must be self-contained.
* **Observability:** Return JSON errors on bad input.
* **Reproducibility:** Same code + GitHub Actions should retrain and rebuild deterministically.

---

## Iteration Plan

### Iteration 1 (v0.1) — Baseline

Simple `StandardScaler` + `LinearRegression`.
Report RMSE on a held-out split.
Ship a working API & Docker image.

### Iteration 2 (v0.2) — Improvement

Try `Ridge` / `RandomForestRegressor` or better preprocessing (feature scaling/selection),
plus calibration of the score if you convert to a “high-risk” flag (e.g., threshold on predicted progression).
Show metric deltas (RMSE; if you add a flag, also precision/recall at a threshold) in `CHANGELOG.md`.

---

## Grading

### CI pipeline quality (3.0)

Runs on PR/push, fails on lint/tests, artifacts uploaded; tag workflow builds image, runs container smoke tests, publishes GitHub Release & GHCR.

### Training & reproducibility (2.0)

Seeds set; env pinned; metrics logged & saved; clear instructions to reproduce locally.

### Docker image quality (2.0)

Self-contained (model baked in), starts quickly, correct port exposed, reasonable size (slim or multi-stage), optional healthcheck.

### Iteration quality & evidence (2.0)

Clear v0.1 → v0.2 improvement (accuracy/latency/size/etc.).
`CHANGELOG.md` shows what changed and why, with side-by-side metrics.

### Documentation & collaboration (1.0)

`README` (exact run commands, sample payload), tidy commit history/PRs.

---

## Acceptance (What I’ll Test)

* I can pull the GitHub Release image (`ghcr.io/<org>/<repo>:v0.1` and `:v0.2`) and run it locally.

* `GET /health` returns:

  ```json
  {"status": "ok", "model_version": "..."}
  ```

* `POST /predict` with a JSON of the diabetes features returns a numeric prediction:

  ```json
  {
    "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
    "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
    "s5": 0.02, "s6": -0.001
  }
  ```

  →

  ```json
  {"prediction": <float>}
  ```

  *(Document your exact field names and response shape).*

* The `v0.2` image shows a justified improvement (metrics + short rationale).

---

## Hand-in

* **GitHub repository URL** (upload the link in a PDF here) — public.

### The Actions tab must show:

* **PR/push workflow:** lint, unit tests, quick training smoke, artifacts.
* **Tag workflow (v*):** builds the Docker image, runs container smoke tests, pushes to GHCR, and publishes a GitHub Release with metrics/changelog.

```
```
