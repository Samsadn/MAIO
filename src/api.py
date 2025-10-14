# FastAPI app: /health, /predict
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

# Import the schemas you defined
from .schemas import DiabetesFeatures, PredictionResponse 

# --- Configuration ---
MODEL_PATH = "artifacts/model.joblib"
MODEL_VERSION = "v0.1"

# Load the model once when the application starts
try:
    # Load the trained scikit-learn pipeline
    MODEL = joblib.load(MODEL_PATH)
except FileNotFoundError:
    # This should only happen if train.py hasn't run
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Run src/train.py first.")

# Initialize FastAPI app
app = FastAPI(
    title="Virtual Diabetes Clinic Triage ML Service",
    description="Predicts short-term disease progression risk score.",
    version=MODEL_VERSION
)

@app.get("/health", response_model=dict)
def health_check():
    """Returns the status of the service and the model version."""
    # Ensure this exact dictionary format is returned
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict", response_model=PredictionResponse)
def predict_progression(data: DiabetesFeatures):
    """
    Predicts the short-term disease progression risk score (continuous).
    """
    try:
        # Convert Pydantic model to a Pandas DataFrame for model input
        # The column order must match the training data exactly
        input_data = pd.DataFrame([data.model_dump()])
        
        # Make the prediction using the loaded scikit-learn pipeline
        prediction_result = MODEL.predict(input_data)[0]
        
        # Return the prediction structured by the response schema
        return PredictionResponse(prediction=float(prediction_result))
        
    except Exception as e:
        # Return a JSON error on bad prediction/internal error (Observability constraint)
        raise HTTPException(status_code=500, detail=f"Internal prediction error: {e}")

# Note: You will need to run this app using uvicorn (see step 3)