"""
api/routes_data.py — REST endpoints for master data management.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List
import io

import state
from pipelines.data_pipeline import run as run_data

router = APIRouter()


class BuildRequest(BaseModel):
    start_date: str = "2018-01-01"
    end_date:   str = "2024-12-31"
    mode:       str = "default"
    sel_indices: List[str] = []
    sel_features: List[str] = ["OHLC", "Returns", "Ratios", "Time"]
    derived_features: List[str] = []


@router.post("/build")
def build_master(req: BuildRequest):
    """Build and cache the master DataFrame."""
    try:
        df = run_data(
            start_date=req.start_date,
            end_date=req.end_date,
            mode=req.mode,
            sel_indices=req.sel_indices or None,
            sel_features=req.sel_features,
            derived_features=req.derived_features,
        )
        return {"rows": len(df), "columns": len(df.columns), "status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/info")
def master_info():
    """Return shape and column names of the cached master DataFrame."""
    if state.MASTER_DF is None:
        raise HTTPException(status_code=404, detail="Master not built yet")
    df = state.MASTER_DF
    return {
        "rows":    len(df),
        "columns": list(df.columns),
        "date_min": str(df["Date"].min().date()),
        "date_max": str(df["Date"].max().date()),
    }


@router.get("/download")
def download_master():
    """Download the cached master CSV."""
    if state.MASTER_DF is None:
        raise HTTPException(status_code=404, detail="Master not built yet")
    buf = io.StringIO()
    state.MASTER_DF.to_csv(buf, index=False)
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=master.csv"},
    )
