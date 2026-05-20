from __future__ import annotations

from typing import List

from pydantic import BaseModel, field_validator


class MonthlyData(BaseModel):
    mois_annee: str  # format "YYYY-MM"
    total_credit: float
    total_debit: float
    nb_transactions: int

    @field_validator("mois_annee")
    @classmethod
    def validate_format(cls, v: str) -> str:
        from datetime import datetime
        datetime.strptime(v, "%Y-%m")
        return v


class PredictRequest(BaseModel):
    user_id: int
    months: List[MonthlyData]

    @field_validator("months")
    @classmethod
    def at_least_three(cls, v: List[MonthlyData]) -> List[MonthlyData]:
        if len(v) < 3:
            raise ValueError("Au moins 3 mois de données sont requis pour faire une prédiction.")
        return sorted(v, key=lambda m: m.mois_annee)


class PredictionResult(BaseModel):
    predicted_credit: float
    predicted_debit: float
    predicted_solde: float
    confidence: str  # "low" | "medium" | "high"
    next_month: str  # "YYYY-MM"


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    version: str = "1.0.0"
