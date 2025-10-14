# tests/test_api_contract.py
from src.schemas import DiabetesFeatures, PredictionResponse

def test_payload_schema_roundtrip():
    payload = dict(
        age=0.02, sex=-0.044, bmi=0.06, bp=-0.03,
        s1=-0.02, s2=0.03, s3=-0.02, s4=0.02, s5=0.02, s6=-0.001
    )
    req = DiabetesFeatures(**payload)
    # Pydantic v2: .model_dump()
    assert req.model_dump() == payload

def test_response_schema():
    res = PredictionResponse(prediction=123.456)
    assert isinstance(res.prediction, float)

