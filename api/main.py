"""FastAPI application exposing the PayLater credit scoring engine.

Run with:
    uvicorn api.main:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException

from .schemas import (
    ApplicantRequest,
    HealthResponse,
    ModelInfo,
    PredictionResponse,
)
from .service import get_service

app = FastAPI(
    title="PayLater Credit Scoring API",
    description="Real-time default-risk scoring for e-commerce PayLater checkout.",
    version="1.0.0",
)


@app.get("/", response_model=HealthResponse)
def root():
    return {"status": "ok", "model": get_service().info()["model"]}


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok", "model": get_service().info()["model"]}


@app.get("/model/info", response_model=ModelInfo)
def model_info():
    return get_service().info()


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ApplicantRequest):
    try:
        return get_service().score(request)
    except Exception as error:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Scoring failed: {error}") from error
