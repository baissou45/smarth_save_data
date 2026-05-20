from __future__ import annotations

import os
import pickle
from calendar import monthrange
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd

from schemas import MonthlyData, PredictionResult

MODELS_DIR = os.getenv(
    "MODELS_DIR",
    os.path.join(os.path.dirname(__file__), "..", "Dossier_smartsave", "models_smartsave"),
)


class Predictor:
    def __init__(self) -> None:
        self._credit_model = None
        self._debit_model = None
        self._feature_scaler = None
        self._target_scaler_credit = None
        self._target_scaler_debit = None
        self._features_list: List[str] = []
        self._seq_len: int = 3
        self.loaded = False

    def load(self) -> None:
        import tensorflow as tf

        scalers_path = os.path.join(MODELS_DIR, "scalers_smartsave.pkl")
        credit_path = os.path.join(MODELS_DIR, "credit_cnn_lstm_attention.keras")
        debit_path = os.path.join(MODELS_DIR, "debit_model_lstm.keras")

        with open(scalers_path, "rb") as f:
            scalers = pickle.load(f)

        self._feature_scaler = scalers["feature_scaler"]
        self._target_scaler_credit = scalers["target_scaler_credit"]
        self._target_scaler_debit = scalers["target_scaler_debit"]
        self._features_list = scalers["features_list"]
        self._seq_len = int(scalers.get("SEQ_LEN", 3))

        self._credit_model = tf.keras.models.load_model(credit_path, compile=False)
        self._debit_model = tf.keras.models.load_model(debit_path, compile=False)

        self.loaded = True

    def _build_features(self, months: List[MonthlyData]) -> np.ndarray:
        """
        Reconstruit les 12 features à partir des agrégats mensuels bruts.
        Les mois sont supposés triés chronologiquement (garanti par le schéma).
        On utilise les 3 derniers mois pour former la séquence.
        """
        df = pd.DataFrame([
            {
                "mois_annee": m.mois_annee,
                "total_credit": float(m.total_credit),
                "total_debit": float(m.total_debit),
                "nb_transactions": int(m.nb_transactions),
            }
            for m in months
        ]).sort_values("mois_annee").reset_index(drop=True)

        df["solde_net"] = df["total_credit"] - df["total_debit"]

        # Lag features
        df["credit_lag1"] = df["total_credit"].shift(1)
        df["debit_lag1"] = df["total_debit"].shift(1)
        df["solde_lag1"] = df["solde_net"].shift(1)

        # Tendance : ratio vs mois m-2
        df["credit_lag2"] = df["total_credit"].shift(2)
        df["debit_lag2"] = df["total_debit"].shift(2)
        df["tendance_credit"] = df["credit_lag1"] / (df["credit_lag2"] + 1)
        df["tendance_debit"] = df["debit_lag1"] / (df["debit_lag2"] + 1)

        # Rolling 3 mois backward-looking (shift(1) pour exclure le mois courant)
        df["moyenne_credit_3m"] = df["total_credit"].shift(1).rolling(3, min_periods=1).mean()
        df["moyenne_debit_3m"] = df["total_debit"].shift(1).rolling(3, min_periods=1).mean()
        df["std_credit_3m"] = df["total_credit"].shift(1).rolling(3, min_periods=1).std()
        df["std_debit_3m"] = df["total_debit"].shift(1).rolling(3, min_periods=1).std()

        # Nettoyage
        cols_fill = [
            "credit_lag1", "debit_lag1", "solde_lag1",
            "tendance_credit", "tendance_debit",
            "moyenne_credit_3m", "moyenne_debit_3m",
            "std_credit_3m", "std_debit_3m",
        ]
        df[cols_fill] = df[cols_fill].fillna(0)
        df.replace([np.inf, -np.inf], 0, inplace=True)

        # Extraire les 12 features dans l'ordre exact du scaler
        sequence = df[self._features_list].values  # shape (n_months, 12)

        # On garde les 3 derniers mois
        sequence = sequence[-self._seq_len:]
        return sequence  # shape (3, 12)

    def _confidence(self, n_months: int) -> str:
        if n_months < 4:
            return "low"
        if n_months < 6:
            return "medium"
        return "high"

    def _next_month_str(self, months: List[MonthlyData]) -> str:
        last = sorted(months, key=lambda m: m.mois_annee)[-1].mois_annee
        y, mo = int(last[:4]), int(last[5:7])
        mo += 1
        if mo > 12:
            mo = 1
            y += 1
        return f"{y:04d}-{mo:02d}"

    def predict(self, months: List[MonthlyData]) -> PredictionResult:
        if not self.loaded:
            raise RuntimeError("Les modèles ne sont pas chargés.")

        sequence = self._build_features(months)  # (3, 12)
        scaled = self._feature_scaler.transform(sequence)  # (3, 12)
        batch = scaled.reshape(1, self._seq_len, len(self._features_list))  # (1, 3, 12)

        pred_credit_scaled = float(self._credit_model.predict(batch, verbose=0)[0, 0])
        pred_debit_scaled = float(self._debit_model.predict(batch, verbose=0)[0, 0])

        pred_credit = float(
            self._target_scaler_credit.inverse_transform([[pred_credit_scaled]])[0, 0]
        )
        pred_debit = float(
            self._target_scaler_debit.inverse_transform([[pred_debit_scaled]])[0, 0]
        )

        return PredictionResult(
            predicted_credit=round(pred_credit, 2),
            predicted_debit=round(pred_debit, 2),
            predicted_solde=round(pred_credit - pred_debit, 2),
            confidence=self._confidence(len(months)),
            next_month=self._next_month_str(months),
        )


predictor = Predictor()
