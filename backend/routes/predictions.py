from datetime import datetime

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    HTTPException,
)

import database
import predictor
import shap_explainer
import alert_engine

from config import get_settings

from utils import (
    parse_and_validate_csv,
    dataframe_to_records,
)

from models import PredictionResult

from logging_config import get_logger


router = APIRouter(
    prefix="/patients",
    tags=["Predictions"],
)

logger = get_logger(__name__)

settings = get_settings()


# ─────────────────────────────────────────────────────────
# RE-RUN PREDICTION
# ─────────────────────────────────────────────────────────

@router.post(
    "/{patient_id}/predict",
    response_model=PredictionResult,
)
async def rerun_prediction(
    patient_id: int,
    file: UploadFile = File(...),
):
    """
    Re-run the prediction pipeline for an existing patient
    with a new CSV upload.

    The new prediction stores:

    - model probability
    - risk level
    - SHAP factors
    - model version
    - exact engineered model input features
    - clinical recommendation

    The feature snapshot is required by the future feedback
    and model-training dataset pipeline.
    """

    # ------------------------------------------------------
    # 1. Verify patient
    # ------------------------------------------------------

    patient = database.get_patient_by_id(
        patient_id
    )

    if not patient:

        raise HTTPException(
            status_code=404,
            detail="Patient not found.",
        )

    # ------------------------------------------------------
    # 2. Parse and validate CSV
    # ------------------------------------------------------

    file_bytes = await file.read()

    df = parse_and_validate_csv(
        file_bytes,
        file.filename,
    )

    # ------------------------------------------------------
    # 3. Overwrite vitals
    # ------------------------------------------------------

    database.save_vitals_batch(
        patient_id,
        dataframe_to_records(df),
    )

    # ------------------------------------------------------
    # 4. Run prediction
    # ------------------------------------------------------

    pred = predictor.predict(
        df
    )

    # ------------------------------------------------------
    # 5. SHAP explanation
    # ------------------------------------------------------

    shap_factors = shap_explainer.explain(
        df
    )

    # ------------------------------------------------------
    # 6. Clinical recommendation
    # ------------------------------------------------------

    recommendation = (
        predictor.clinical_recommendation(
            pred["risk_level"]
        )
    )

    # ------------------------------------------------------
    # 7. Save prediction
    # ------------------------------------------------------

    prediction_id = database.save_prediction(
        patient_id=patient_id,

        probability=pred["probability"],

        risk_level=pred["risk_level"],

        shap_factors=shap_factors,

        # NEW
        input_features=pred.get(
            "features",
            {},
        ),

        # NEW
        model_version=settings.model_version,

        # NEW
        clinical_recommendation=recommendation,
    )

    # ------------------------------------------------------
    # 8. Alert check
    # ------------------------------------------------------

    alert_engine.check_and_trigger(
        patient_id=patient_id,

        patient_name=patient["name"],

        bed_number=patient["bed_number"],

        new_risk_level=pred["risk_level"],

        probability=pred["probability"],
    )

    # ------------------------------------------------------
    # 9. Response
    # ------------------------------------------------------

    now = datetime.utcnow().isoformat()

    logger.info(
        "Re-run prediction patient_id=%d "
        "prediction_id=%d risk=%s prob=%.4f",
        patient_id,
        prediction_id,
        pred["risk_level"],
        pred["probability"],
    )

    return {
        "patient_id":
            patient_id,

        "prediction_id":
            prediction_id,

        "sepsis_probability":
            pred["probability"],

        "risk_level":
            pred["risk_level"],

        "shap_factors":
            shap_factors,

        "predicted_at":
            now,

        "model_version":
            settings.model_version,

        "clinical_recommendation":
            recommendation,
    }