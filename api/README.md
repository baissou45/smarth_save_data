# SmartSave Prediction API

Microservice FastAPI qui expose les modèles ML de prévision financière de SmartSave.  
Il prédit les **entrées** et **sorties** mensuelles d'un utilisateur à partir de son historique de transactions.

---

## Architecture

```
Flutter app  →  POST /predict  →  FastAPI  →  .keras models + scalers.pkl  →  JSON
```

Le service tourne indépendamment du backend Laravel, sur le **port 8001**.

---

## Modèles utilisés

| Modèle | Fichier | Architecture | Prédit |
|--------|---------|-------------|--------|
| Crédit | `credit_cnn_lstm_attention.keras` | CNN-LSTM + Attention | Entrées du mois suivant |
| Débit | `debit_model_lstm.keras` | LSTM 2 couches | Sorties du mois suivant |

Les modèles attendent une **fenêtre de 3 mois** (12 features par mois, StandardScaler).

---

## Installation

```bash
cd smarth_save_data/api

# Créer un environnement virtuel (recommandé)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Variable d'environnement

Copier `.env.example` en `.env` et ajuster le chemin des modèles si nécessaire :

```bash
cp .env.example .env
```

```env
MODELS_DIR=D:/taf/data/smarth_save_data/Dossier_smartsave/models_smartsave
```

Par défaut (sans `.env`), le service cherche les modèles dans `../Dossier_smartsave/models_smartsave/` relativement au dossier `api/`.

---

## Démarrage

```bash
python -m uvicorn main:app --port 8001 --reload
```

> Sur Windows, `uvicorn` n'est généralement pas dans le PATH — utiliser `python -m uvicorn`.

Le service charge les deux modèles `.keras` et les scalers au démarrage.  
Première requête légèrement plus lente (warm-up TensorFlow).

---

## Endpoints

### `GET /`
Info de version.

```json
{ "service": "SmartSave Prediction API", "version": "1.0.0", "status": "running" }
```

---

### `GET /health`
Vérifie que les modèles sont bien chargés.

```json
{ "status": "ok", "models_loaded": true, "version": "1.0.0" }
```

---

### `POST /predict`
Prédit les entrées et sorties du mois suivant.

**Body :**
```json
{
  "user_id": 1,
  "months": [
    { "mois_annee": "2026-02", "total_credit": 800.0, "total_debit": 700.0, "nb_transactions": 25 },
    { "mois_annee": "2026-03", "total_credit": 850.0, "total_debit": 720.0, "nb_transactions": 28 },
    { "mois_annee": "2026-04", "total_credit": 900.0, "total_debit": 680.0, "nb_transactions": 30 }
  ]
}
```

> **Minimum 3 mois requis**, triés chronologiquement.  
> `user_id` est accepté mais non utilisé (les données sont passées directement par Flutter).

**Réponse :**
```json
{
  "predicted_credit": 872.50,
  "predicted_debit":  714.30,
  "predicted_solde":  158.20,
  "confidence":       "medium",
  "next_month":       "2026-05"
}
```

| Champ | Description |
|-------|-------------|
| `predicted_credit` | Entrées estimées en € pour le mois suivant |
| `predicted_debit` | Sorties estimées en € pour le mois suivant |
| `predicted_solde` | `credit - debit` estimé |
| `confidence` | `"low"` (<4 mois) · `"medium"` (4-5 mois) · `"high"` (6+ mois) |
| `next_month` | Mois prédit au format `YYYY-MM` |

**Codes d'erreur :**

| Code | Raison |
|------|--------|
| 422 | Moins de 3 mois fournis ou format `mois_annee` invalide |
| 503 | Modèles non chargés (démarrage en cours) |
| 500 | Erreur interne lors de la prédiction |

---

## Pipeline d'inférence

Pour chaque requête :

1. **Agrégation** — Les mois sont triés par date
2. **Feature engineering** — Calcul des 12 features (lags, rolling 3 mois, tendances)
3. **Scaling** — `StandardScaler.transform()` sur la fenêtre de 3 mois → shape `(3, 12)`
4. **Prédiction** — `model.predict(batch.reshape(1, 3, 12))` → scalaire normalisé
5. **Inverse transform** — Retour en euros via `target_scaler.inverse_transform()`

Les 12 features (ordre exact du scaler) :

```
total_credit, total_debit, credit_lag1, debit_lag1,
moyenne_credit_3m, moyenne_debit_3m, std_credit_3m, std_debit_3m,
solde_lag1, tendance_credit, tendance_debit, nb_transactions
```

---

## Intégration Flutter

Le service est appelé depuis `lib/services/api_prediction_service.dart`.  
La classe `ApiPredictionService` :
- Agrège les transactions du provider en données mensuelles
- Appelle `POST /predict` sur `http://10.0.2.2:8001` (émulateur Android)
- Retourne un `PredictionModel` (ou `null` si < 3 mois de données)

Le résultat est affiché sur le dashboard dans la section **"Prévisions"**, respectant le toggle de visibilité des montants.

---

## Structure du projet

```
api/
├── main.py          # App FastAPI, endpoints, lifespan
├── predictor.py     # Chargement modèles + pipeline d'inférence
├── schemas.py       # Pydantic models (request / response)
├── requirements.txt # Dépendances Python
├── .env.example     # Template de configuration
└── README.md        # Ce fichier
```
