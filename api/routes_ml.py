"""
api/routes_ml.py — REST endpoints for ML training and explainability.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

import state
from pipelines.ml_pipeline import run as run_ml
from services.ml_service import compute_shap, compute_lime

router = APIRouter()


class TrainRequest(BaseModel):
    model_name:  str   = "Random Forest"
    train_start: str   = "2018-01-01"
    train_end:   str   = "2022-12-31"
    test_start:  str   = "2023-01-01"
    test_end:    str   = "2024-12-31"
    prob_cutoff: float = 0.5
    run_shap:    bool  = False
    run_lime:    bool  = False


@router.post("/train")
def train(req: TrainRequest):
    """Train a model and return performance metrics."""
    try:
        results = run_ml(**req.dict())
        return {
            "model":    req.model_name,
            "accuracy": round(results["accuracy"], 4),
            "roc_auc":  round(float(results["roc_auc"]), 4),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/shap")
def shap_endpoint():
    """Return top-20 mean |SHAP| values for the last trained model."""
    if state.LAST_PIPELINE is None:
        raise HTTPException(status_code=404, detail="No model trained yet")
    try:
        shap_df, _, _ = compute_shap(
            state.LAST_PIPELINE, state.LAST_X_TRAIN,
            state.LAST_X_TEST, state.LAST_MODEL_NAME,
        )
        return shap_df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/lime")
def lime_endpoint(sample_idx: int = 0):
    """Return LIME weights for a specific test sample."""
    if state.LAST_PIPELINE is None:
        raise HTTPException(status_code=404, detail="No model trained yet")
    try:
        lime_df, _ = compute_lime(
            state.LAST_PIPELINE, state.LAST_X_TRAIN,
            state.LAST_X_TEST, sample_idx=sample_idx,
        )
        return lime_df.to_dict(orient="records")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
