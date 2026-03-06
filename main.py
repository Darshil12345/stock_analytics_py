"""
main.py — FastAPI server.

Mounts the Dash app via WSGIMiddleware and exposes REST API routes under /api/.
Run with:  uvicorn main:fastapi_app --reload --host 0.0.0.0 --port 8050
"""
from fastapi import FastAPI
from starlette.middleware.wsgi import WSGIMiddleware

from dashboard import create_dash_app
from api.routes_data import router as data_router
from api.routes_ml import router as ml_router
from api.routes_news import router as news_router

# ── FastAPI instance ──────────────────────────────────────────────────────────
fastapi_app = FastAPI(
    title="Stock Analytics API",
    description="REST endpoints powering the Dash dashboard",
    version="2.0.0",
)

# ── REST routes ───────────────────────────────────────────────────────────────
fastapi_app.include_router(data_router, prefix="/api/data",  tags=["Data"])
fastapi_app.include_router(ml_router,   prefix="/api/ml",    tags=["ML"])
fastapi_app.include_router(news_router, prefix="/api/news",  tags=["News"])

# ── Dash app (mounted last so /api/* is matched first) ────────────────────────
_dash_app = create_dash_app()
fastapi_app.mount("/", WSGIMiddleware(_dash_app.server))


# ── Dev entry-point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:fastapi_app", host="0.0.0.0", port=8050, reload=True)
