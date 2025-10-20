````markdown
# Assignment 3 – MLOps  
## Virtual Diabetes Clinic Triage

### 📘 Context
Söderstad Hospital runs a virtual diabetes clinic where nurses review hundreds of weekly patient check-ins (vitals, labs, and lifestyle notes) to decide who needs a follow-up call.  
These reviews are currently **manual and time-consuming**.

### 🎯 Goal
The objective is to build a **machine learning service** that predicts a patient's short-term risk of disease progression and outputs a **continuous risk score**.  
The clinic will use this score to prioritize patients who may need faster intervention.

All components — training, packaging, testing, and releasing — are automated through **GitHub Actions** CI/CD pipelines.

---

### 📊 Data
The project uses the open **scikit-learn Diabetes regression dataset** as a proxy for de-identified electronic health record (EHR) data.

```python
from sklearn.datasets import load_diabetes

Xy = load_diabetes(as_frame=True)
X = Xy.frame.drop(columns=["target"])
y = Xy.frame["target"]  # progression index (higher = worse)
````

The dataset mimics features like blood pressure, BMI, and cholesterol ratios that can influence diabetes progression.

---

### ⚙️ Pipeline Overview

| Stage                               | Description                                                                                                                                                    |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Training (`src/train.py`)**       | Loads dataset, splits data, trains a scikit-learn pipeline (`StandardScaler` + `LinearRegression` or `Ridge`), computes RMSE, and saves the model and metrics. |
| **Serving (`src/api.py`)**          | FastAPI service exposing two endpoints: `/health` and `/predict`.                                                                                              |
| **Schemas (`src/schemas.py`)**      | Defines input/output Pydantic models for API validation.                                                                                                       |
| **Configuration (`src/config.py`)** | Centralized configuration for reproducibility and artifact paths.                                                                                              |
| **CI/CD Workflows**                 | Automates linting, testing, training, Docker build, smoke tests, and GHCR release.                                                                             |

---

### 🧠 Model Versions

| Version    | Model                                 | RMSE          | Notes                                |
| ---------- | ------------------------------------- | ------------- | ------------------------------------ |
| **v0.1.0** | `StandardScaler` + `LinearRegression` | *e.g., 54.82* | Baseline model.                      |
| **v0.2.0** | `StandardScaler` + `Ridge(alpha=1.0)` | *e.g., 52.47* | Improved regularization, lower RMSE. |

Each model is saved in `/artifacts` and baked into the Docker image at release.

---

### 🧪 API Endpoints

#### **GET /health**

Returns the service status and model version.

**Example Response**

```json
{"status": "ok", "model_version": "v0.2.0"}
```

#### **POST /predict**

Predicts short-term disease progression based on numeric input features.

**Example Request**

```json
{
  "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
  "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
}
```

**Example Response**

```json
{"prediction": 156.78}
```

---

### 🧰 Local Development

#### **Train the model**

```bash
python -m src.train
```

#### **Run the API**

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

#### **Test the API**

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":0.02,"sex":-0.044,"bmi":0.06,"bp":-0.03,"s1":-0.02,"s2":0.03,"s3":-0.02,"s4":0.02,"s5":0.02,"s6":-0.001}'
```

---

### 🐳 Docker Usage

#### **Build and run locally**

```bash
docker build -t diabetes-triage .
docker run -p 8000:8000 diabetes-triage
```

#### **Pull released image (from GHCR)**

```bash
docker pull ghcr.io/samsadn/maio:v0.2.1
docker run -p 8000:8000 ghcr.io/samsadn/maio:v0.2.1
```

---

### 🔄 CI/CD Summary

| Workflow                    | Trigger      | Purpose                                                                             |
| --------------------------- | ------------ | ----------------------------------------------------------------------------------- |
| **CI (`ci.yml`)**           | On push / PR | Runs linting (`ruff`), pytest, and uploads artifacts.                               |
| **Release (`release.yml`)** | On tag (v*)  | Retrains, builds Docker, smoke-tests, pushes to GHCR, and publishes GitHub Release. |

---

### 🧮 Reproducibility

* Fixed random seed (`42`)
* Versioned artifacts in `/artifacts`
* Pinned dependencies in `requirements.txt`
* Deterministic CI build & release pipelines

---

### 🧾 References

* scikit-learn Diabetes Dataset Documentation
* [FastAPI](https://fastapi.tiangolo.com/)
* [GitHub Actions](https://docs.github.com/en/actions)
* [Docker](https://docs.docker.com/)

---