import csv
import io
import math

from datetime import datetime

from fastapi import (
    APIRouter,
    Depends,
    Header,
    HTTPException,
)

from fastapi.responses import StreamingResponse

from sqlalchemy.exc import IntegrityError

import database

from config import get_settings

from models import (
    PredictionFeedbackCreate,
    PredictionFeedbackResponse,
    FeedbackStats,
)


router = APIRouter(
    prefix="/feedback",
    tags=["Prediction Feedback"],
)

settings = get_settings()


# ─────────────────────────────────────────────────────────
# AUTHORIZATION
# ─────────────────────────────────────────────────────────

def require_feedback_access(
    x_feedback_api_key: str | None = Header(
        default=None
    ),
    x_user_id: str | None = Header(
        default=None
    ),
) -> str:
    """
    Basic authorization layer for the feedback workflow.

    The frontend must send:

        X-Feedback-API-Key
        X-User-Id
    """

    if not settings.feedback_api_key:

        raise HTTPException(
            status_code=503,
            detail=(
                "Feedback authorization is not configured."
            ),
        )

    if (
        x_feedback_api_key
        != settings.feedback_api_key
    ):

        raise HTTPException(
            status_code=403,
            detail=(
                "Not authorized to submit "
                "or export feedback."
            ),
        )

    if (
        not x_user_id
        or not x_user_id.strip()
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "X-User-Id header is required."
            ),
        )

    return x_user_id.strip()


# ─────────────────────────────────────────────────────────
# FEEDBACK VALIDATION
# ─────────────────────────────────────────────────────────

def _validate_feedback(
    prediction: dict,
    payload: PredictionFeedbackCreate,
) -> tuple[str, str]:
    """
    Validate whether feedback can become a training
    dataset record.

    Returns:

        ("ELIGIBLE", reason)
        ("PENDING", reason)
        ("REJECTED", reason)
    """

    # ------------------------------------------------------
    # Prediction must exist
    # ------------------------------------------------------

    if not prediction:

        return (
            "REJECTED",
            "Referenced prediction does not exist.",
        )

    # ------------------------------------------------------
    # Model input snapshot must exist
    # ------------------------------------------------------

    features = (
        prediction.get(
            "input_features"
        )
        or {}
    )

    if not features:

        return (
            "REJECTED",
            (
                "Prediction is missing the "
                "model input feature snapshot."
            ),
        )

    # ------------------------------------------------------
    # Every feature must be numeric and finite
    # ------------------------------------------------------

    for feature_name, value in features.items():

        if value is None:

            return (
                "REJECTED",
                (
                    f"Feature '{feature_name}' "
                    "is missing."
                ),
            )

        if not isinstance(
            value,
            (int, float),
        ):

            return (
                "REJECTED",
                (
                    f"Feature '{feature_name}' "
                    "is not numeric."
                ),
            )

        if not math.isfinite(
            float(value)
        ):

            return (
                "REJECTED",
                (
                    f"Feature '{feature_name}' "
                    "contains a non-finite value."
                ),
            )

    # ------------------------------------------------------
    # Probability must be valid
    # ------------------------------------------------------

    probability = prediction.get(
        "sepsis_probability"
    )

    if probability is None:

        return (
            "REJECTED",
            "Prediction probability is missing.",
        )

    if not math.isfinite(
        float(probability)
    ):

        return (
            "REJECTED",
            (
                "Prediction probability "
                "is not finite."
            ),
        )

    if not (
        0.0
        <= float(probability)
        <= 1.0
    ):

        return (
            "REJECTED",
            (
                "Prediction probability "
                "must be between 0 and 1."
            ),
        )

    # ------------------------------------------------------
    # Risk level
    # ------------------------------------------------------

    if prediction.get(
        "risk_level"
    ) not in {
        "LOW",
        "MEDIUM",
        "HIGH",
    }:

        return (
            "REJECTED",
            "Prediction risk level is invalid.",
        )

    # ------------------------------------------------------
    # Actual outcome not known yet
    # ------------------------------------------------------

    if payload.actual_outcome == "UNKNOWN":

        return (
            "PENDING",
            (
                "Actual clinical outcome is not "
                "available yet. Record it later "
                "before dataset preparation."
            ),
        )

    # ------------------------------------------------------
    # Actual outcome is known
    # ------------------------------------------------------

    if payload.actual_outcome in {
        "SEPSIS",
        "NO_SEPSIS",
    }:

        if (
            not payload.outcome_at
            or not payload.outcome_at.strip()
        ):

            return (
                "REJECTED",
                (
                    "Outcome timestamp is required "
                    "when the actual outcome is known."
                ),
            )

        return (
            "ELIGIBLE",
            (
                "Validated: prediction, model inputs, "
                "model version and clinical outcome "
                "are present."
            ),
        )

    return (
        "REJECTED",
        "Unsupported clinical outcome.",
    )


# ─────────────────────────────────────────────────────────
# SUBMIT FEEDBACK
# ─────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=PredictionFeedbackResponse,
    status_code=201,
)
def submit_feedback(
    payload: PredictionFeedbackCreate,
    submitted_by: str = Depends(
        require_feedback_access
    ),
):
    """
    Submit clinician feedback for a prediction.
    """

    # ------------------------------------------------------
    # Get original prediction
    # ------------------------------------------------------

    prediction = database.get_prediction_by_id(
        payload.prediction_id
    )

    if not prediction:

        raise HTTPException(
            status_code=404,
            detail="Prediction not found.",
        )

    # ------------------------------------------------------
    # Validate feedback
    # ------------------------------------------------------

    status, reason = _validate_feedback(
        prediction,
        payload,
    )

    # ------------------------------------------------------
    # Store feedback
    # ------------------------------------------------------

    try:

        result = database.create_feedback(

            prediction_id=
                payload.prediction_id,

            patient_id=
                prediction["patient_id"],

            submitted_by=
                submitted_by,

            prediction_assessment=
                payload.prediction_assessment,

            recommendation_assessment=
                payload.recommendation_assessment,

            actual_outcome=
                payload.actual_outcome,

            outcome_at=
                payload.outcome_at,

            outcome_notes=
                payload.outcome_notes,

            validation_status=
                status,

            validation_reason=
                reason,
        )

    except IntegrityError:

        raise HTTPException(
            status_code=409,
            detail=(
                "Feedback from this user for "
                "this prediction already exists. "
                "Duplicate feedback was not stored."
            ),
        )

    return result


# ─────────────────────────────────────────────────────────
# UPDATE FEEDBACK
# ─────────────────────────────────────────────────────────

@router.patch(
    "/{feedback_id}",
    response_model=PredictionFeedbackResponse,
)
def update_prediction_feedback(
    feedback_id: int,

    payload: PredictionFeedbackCreate,

    submitted_by: str = Depends(
        require_feedback_access
    ),
):
    """
    Update previously submitted feedback.

    This is primarily used when an initially UNKNOWN
    clinical outcome becomes available later.
    """

    existing = database.get_feedback(
        feedback_id
    )

    if not existing:

        raise HTTPException(
            status_code=404,
            detail="Feedback not found.",
        )

    # ------------------------------------------------------
    # Ownership
    # ------------------------------------------------------

    if (
        existing["submitted_by"]
        != submitted_by
    ):

        raise HTTPException(
            status_code=403,
            detail=(
                "Only the original feedback "
                "submitter can update this record."
            ),
        )

    # ------------------------------------------------------
    # Don't modify records already exported
    # ------------------------------------------------------

    if existing.get(
        "dataset_exported_at"
    ):

        raise HTTPException(
            status_code=409,
            detail=(
                "Feedback has already been exported "
                "and cannot be modified."
            ),
        )

    # ------------------------------------------------------
    # Prediction must match original feedback
    # ------------------------------------------------------

    if (
        payload.prediction_id
        != existing["prediction_id"]
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "prediction_id cannot be changed "
                "when updating feedback."
            ),
        )

    prediction = database.get_prediction_by_id(
        existing["prediction_id"]
    )

    if not prediction:

        raise HTTPException(
            status_code=404,
            detail=(
                "Original prediction no longer exists."
            ),
        )

    # ------------------------------------------------------
    # Revalidate
    # ------------------------------------------------------

    status, reason = _validate_feedback(
        prediction,
        payload,
    )

    # ------------------------------------------------------
    # Update
    # ------------------------------------------------------

    result = database.update_feedback(

        feedback_id=feedback_id,

        prediction_assessment=
            payload.prediction_assessment,

        recommendation_assessment=
            payload.recommendation_assessment,

        actual_outcome=
            payload.actual_outcome,

        outcome_at=
            payload.outcome_at,

        outcome_notes=
            payload.outcome_notes,

        validation_status=
            status,

        validation_reason=
            reason,
    )

    return result


# ─────────────────────────────────────────────────────────
# FEEDBACK STATS
# ─────────────────────────────────────────────────────────

@router.get(
    "/stats",
    response_model=FeedbackStats,
)
def feedback_stats(
    _: str = Depends(
        require_feedback_access
    ),
):
    """
    Return feedback preparation statistics.
    """

    return database.get_feedback_stats()


# ─────────────────────────────────────────────────────────
# TRAINING DATASET EXPORT
# ─────────────────────────────────────────────────────────

@router.post(
    "/dataset/export",
)
def export_training_dataset(
    _: str = Depends(
        require_feedback_access
    ),
):
    """
    Export newly eligible feedback as a CSV training dataset.

    IMPORTANT:
    This endpoint only prepares/exports the dataset.

    It does NOT:
        - train the model
        - replace the production model
        - deploy a new model
    """

    rows = (
        database.get_eligible_feedback_for_dataset(
            include_exported=False
        )
    )

    if not rows:

        raise HTTPException(
            status_code=404,
            detail=(
                "No new eligible feedback "
                "is available for export."
            ),
        )

    output = io.StringIO()

    writer = None

    feedback_ids = []

    # ------------------------------------------------------
    # Build CSV
    # ------------------------------------------------------

    for item in rows:

        prediction = item[
            "prediction"
        ]

        feedback = item[
            "feedback"
        ]

        features = (
            prediction.get(
                "input_features"
            )
            or {}
        )

        row = {
            # ----------------------------------------------
            # Original model features
            # ----------------------------------------------

            **features,

            # ----------------------------------------------
            # Prediction metadata
            # ----------------------------------------------

            "prediction_id":
                prediction["id"],

            "patient_id":
                prediction["patient_id"],

            "model_version":
                prediction["model_version"],

            "model_probability":
                prediction[
                    "sepsis_probability"
                ],

            "model_risk_level":
                prediction[
                    "risk_level"
                ],

            "predicted_at":
                prediction[
                    "predicted_at"
                ],

            # ----------------------------------------------
            # Human feedback
            # ----------------------------------------------

            "prediction_assessment":
                feedback[
                    "prediction_assessment"
                ],

            "recommendation_assessment":
                feedback[
                    "recommendation_assessment"
                ],

            # ----------------------------------------------
            # Training target
            # ----------------------------------------------

            "label":
                (
                    1
                    if feedback[
                        "actual_outcome"
                    ] == "SEPSIS"
                    else 0
                ),

            "actual_outcome":
                feedback[
                    "actual_outcome"
                ],

            "outcome_at":
                feedback[
                    "outcome_at"
                ],

            "outcome_notes":
                feedback[
                    "outcome_notes"
                ],

            # ----------------------------------------------
            # Feedback record
            # ----------------------------------------------

            "feedback_id":
                feedback["id"],

            "feedback_submitted_by":
                feedback["submitted_by"],

            "feedback_created_at":
                feedback["created_at"],
        }

        if writer is None:

            writer = csv.DictWriter(
                output,
                fieldnames=list(
                    row.keys()
                ),
            )

            writer.writeheader()

        writer.writerow(row)

        feedback_ids.append(
            feedback["id"]
        )

    # ------------------------------------------------------
    # Mark records as exported
    # ------------------------------------------------------

    database.mark_feedback_exported(
        feedback_ids
    )

    output.seek(0)

    filename = (
        "pranrakshak_training_dataset_"
        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    return StreamingResponse(
        iter([
            output.getvalue()
        ]),

        media_type="text/csv",

        headers={
            "Content-Disposition":
                (
                    'attachment; '
                    f'filename="{filename}"'
                ),
        },
    )