from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from predictor import predictor
from schemas import HealthResponse, PredictRequest, PredictionResult


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()
    yield


app = FastAPI(
    title="SmartSave Prediction API",
    description="Prévision d'entrées/sorties financières via modèles LSTM/CNN",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://10.0.2.2:8000",
        "http://localhost:8000",
        "http://10.0.2.2:8001",
        "http://localhost:8001",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
def root():
    return {"service": "SmartSave Prediction API", "version": "1.0.0", "status": "running"}


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if predictor.loaded else "degraded",
        models_loaded=predictor.loaded,
    )


@app.post("/predict", response_model=PredictionResult)
def predict(body: PredictRequest) -> PredictionResult:
    if not predictor.loaded:
        raise HTTPException(status_code=503, detail="Les modèles ML ne sont pas encore chargés.")
    try:
        return predictor.predict(body.months)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la prédiction : {e}")
