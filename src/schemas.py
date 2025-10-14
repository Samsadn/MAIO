# Pydantic request/response schemas
from pydantic import BaseModel, Field

# Input Schema for the POST /predict endpoint
class DiabetesFeatures(BaseModel):
    # These names and fields match the diabetes dataset columns
    age: float = Field(..., description="Age in normalized units")
    sex: float = Field(..., description="Sex in normalized units")
    bmi: float = Field(..., description="Body Mass Index")
    bp: float = Field(..., description="Blood Pressure")
    s1: float = Field(..., description="Total serum cholesterol (TC)")
    s2: float = Field(..., description="Low-density lipoproteins (LDL)")
    s3: float = Field(..., description="High-density lipoproteins (HDL)")
    s4: float = Field(..., description="Thyroid stimulating hormone (TSH)")
    s5: float = Field(..., description="Lamotrigine (LMT)")
    s6: float = Field(..., description="Blood sugar")
    
    # Enable example generation for documentation
    class Config:
        json_schema_extra = {
            "example": {
                "age": 0.02,
                "sex": -0.044,
                "bmi": 0.06,
                "bp": -0.03,
                "s1": -0.02,
                "s2": 0.03,
                "s3": -0.02,
                "s4": 0.02,
                "s5": 0.02,
                "s6": -0.001
            }
        }

# Output Schema for the POST /predict endpoint
class PredictionResponse(BaseModel):
    prediction: float = Field(..., description="Predicted short-term disease progression index (higher = worse risk).")