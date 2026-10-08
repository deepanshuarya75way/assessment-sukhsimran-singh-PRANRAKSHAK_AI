import logging
import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from config import get_settings
from database import (
    init_db,
    get_all_patients,
    create_patient,
    save_prediction,
    save_vitals_batch,
)
from logging_config import setup_logging, get_logger
import predictor
import shap_explainer

from routes import patients, predictions, alerts, copilot, feedback


# ─── Logging ────────────────────────────────────────────────────────────────

setup_logging()
logger: logging.Logger = get_logger(name=__name__)

settings = get_settings()


# ─── Demo patient seeding ────────────────────────────────────────────────────

def seed_demo_patients() -> None:
    """Seed demo patients from demo_patients.json if no patients exist yet."""

    existing_patients: list[dict[str, Any]] = get_all_patients()

    if existing_patients:
        logger.info(msg="Demo patients already exist, skipping seeding.")
        return

    demo_patients_path: str = os.path.join(
        os.path.dirname(__file__),
        "demo_patients.json",
    )

    if not os.path.exists(path=demo_patients_path):
        logger.warning(msg="demo_patients.json not found, skipping seeding.")
        return

    with open(file=demo_patients_path, mode="r", encoding="utf-8") as f:
        demo_patients = json.load(fp=f)

    for demo_patient in demo_patients:
        try:
            # Create patient
            patient_id: dict[str, Any] = create_patient(
                name=demo_patient["name"],
                bed_number=demo_patient["bed_number"],
                age=demo_patient["age"],
                gender=demo_patient["gender"],
            )

            logger.info(
                msg="Created demo patient: %s (ID: %d)",
                *(
                    demo_patient["name"],
                    patient_id,
                ),
            )

            # Save vitals
            if "vitals" in demo_patient:
                # Map the vitals to the format save_vitals_batch expects
                vitals_rows: list[Any] = []

                for v in demo_patient["vitals"]:
                    vitals_rows.append(
                        {
                            "Hour": v.get("hour"),
                            "HR": v.get("hr"),
                            "O2Sat": v.get("o2sat"),
                            "Temp": v.get("temp"),
                            "SBP": v.get("sbp"),
                            "MAP": v.get("map_val"),
                            "Resp": v.get("resp"),
                            "WBC": v.get("wbc"),
                            "Creatinine": v.get("creatinine"),
                            "Glucose": v.get("glucose"),
                            "Age": v.get("age"),
                            "ICULOS": v.get("iculos"),
                        }
                    )

                save_vitals_batch(patient_id, vitals_rows)

                logger.info(
                    msg="Saved vitals for demo patient ID: %d",
                    *(
                        patient_id,
                    ),
                )

            # Save prediction with SHAP factors
            if (
                "latest_probability" in demo_patient
                and "shap_factors" in demo_patient
            ):
                save_prediction(
                    patient_id=patient_id,
                    probability=demo_patient["latest_probability"],
                    risk_level=demo_patient["latest_risk_level"],
                    shap_factors=demo_patient["shap_factors"],
                )

                logger.info(
                    msg="Saved prediction for demo patient ID: %d",
                    *(
                        patient_id,
                    ),
                )

        except Exception as exc:
            logger.error(
                msg="Failed to seed demo patient %s: %s",
                *(
                    demo_patient.get("name"),
                    str(exc),
                ),
            )


# ─── Lifespan: startup + shutdown ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncGenerator[None, Any, None]:
    """
    Startup: initialise DB, load ML model and SHAP explainer, seed demo patients.
    Shutdown: log clean exit.
    """

    logger.info(msg="PranRakshak backend starting up...")

    try:
        init_db()
        seed_demo_patients()
    except Exception as exc:
        logger.warning(
            msg="Database initialization failed: %s",
            *(
                exc,
            ),
        )

    try:
        predictor.load_model()
    except Exception as exc:
        logger.warning(
            msg=(
                "Model loading failed - predictions will fall back to the "
                "model available in predictor.py: %s"
            ),
            *(
                exc,
            ),
        )

    try:
        shap_explainer.load_explainer(
            pipeline=predictor.pipeline,
        )
    except Exception as exc:
        logger.warning(
            msg="SHAP explainer failed to load: %s",
            *(
                exc,
            ),
        )

    logger.info(
        msg="Startup complete. ENV=%s FRONTEND=%s",
        *(
            settings.env,
            settings.frontend_origin,
        ),
    )

    yield

    logger.info(msg="PranRakshak backend shutting down.")


# ─── App ────────────────────────────────────────────────────────────────────

app: FastAPI = FastAPI(
    title="PranRakshak API",
    description=(
        "Intelligent ICU patient risk monitoring and decision support system."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─── CORS ───────────────────────────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://pranrakshak-ai.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Global exception handler ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    logger.error(
        msg="Unhandled exception on %s %s: %s",
        *(
            request.method,
            request.url.path,
            exc,
        ),
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error.",
            "detail": str(exc) if settings.env == "development" else None,
        },
    )


# ─── Health check ───────────────────────────────────────────────────────────

@app.get(path="/health", tags=["system"])
def health_check() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": predictor._pipeline is not None,
        "shap_loaded": shap_explainer._explainer is not None,
        "env": settings.env,
    }


# ─── Routes ─────────────────────────────────────────────────────────────────

app.include_router(patients.router)
app.include_router(predictions.router)
app.include_router(alerts.router)
app.include_router(copilot.router)
app.include_router(feedback.router)