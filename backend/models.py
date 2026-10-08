from datetime import datetime
from typing import Optional

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)


# ─── Shared ───────────────────────────────────────────────────────────────────

class ShapFactor(BaseModel):
    feature: str
    impact: float
    direction: str  # increases_risk | decreases_risk


# ─── Patient ──────────────────────────────────────────────────────────────────

class PatientCreate(BaseModel):

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    bed_number: str = Field(
        ...,
        min_length=1,
        max_length=20,
    )

    age: int = Field(
        ...,
        ge=0,
        le=120,
    )

    gender: str = Field(
        ...,
        pattern="^(Male|Female|Other)$",
    )


class PatientSummary(BaseModel):

    id: int

    name: str

    bed_number: str

    age: int

    gender: str

    created_at: str

    latest_probability: Optional[float] = None

    latest_risk_level: Optional[str] = None

    latest_predicted_at: Optional[str] = None

    # NEW
    latest_prediction_id: Optional[int] = None

    # NEW
    model_version: Optional[str] = None


class VitalsRow(BaseModel):

    hour: Optional[float] = None

    hr: Optional[float] = None

    o2sat: Optional[float] = None

    temp: Optional[float] = None

    sbp: Optional[float] = None

    map_val: Optional[float] = None

    resp: Optional[float] = None

    wbc: Optional[float] = None

    creatinine: Optional[float] = None

    glucose: Optional[float] = None

    age: Optional[float] = None

    iculos: Optional[float] = None

    row_index: int


class PatientDetail(
    PatientSummary
):

    shap_factors: Optional[
        list[ShapFactor]
    ] = None

    vitals: list[VitalsRow] = []

    # NEW
    clinical_recommendation: Optional[str] = None


# ─── Prediction ───────────────────────────────────────────────────────────────

class PredictionResult(BaseModel):

    patient_id: int

    sepsis_probability: float

    risk_level: str

    shap_factors: list[ShapFactor]

    predicted_at: str

    # NEW
    model_version: Optional[str] = None

    # NEW
    clinical_recommendation: Optional[str] = None

    # NEW
    prediction_id: Optional[int] = None


# ─── Prediction Feedback ──────────────────────────────────────────────────────

class PredictionFeedbackCreate(BaseModel):
    """
    Clinician feedback submitted against an existing prediction.
    """

    prediction_id: int = Field(
        ...,
        ge=1,
    )

    prediction_assessment: str

    recommendation_assessment: str

    actual_outcome: str

    outcome_at: Optional[str] = None

    outcome_notes: Optional[str] = Field(
        default=None,
        max_length=2000,
    )

    # ------------------------------------------------------
    # Prediction assessment validation
    # ------------------------------------------------------

    @field_validator(
        "prediction_assessment"
    )
    @classmethod
    def validate_prediction_assessment(
        cls,
        value: str,
    ) -> str:

        value = value.strip().upper()

        allowed = {
            "CORRECT",
            "INCORRECT",
            "UNCERTAIN",
        }

        if value not in allowed:

            raise ValueError(
                "prediction_assessment must be "
                "CORRECT, INCORRECT, or UNCERTAIN."
            )

        return value

    # ------------------------------------------------------
    # Recommendation assessment validation
    # ------------------------------------------------------

    @field_validator(
        "recommendation_assessment"
    )
    @classmethod
    def validate_recommendation_assessment(
        cls,
        value: str,
    ) -> str:

        value = value.strip().upper()

        allowed = {
            "APPROPRIATE",
            "INAPPROPRIATE",
            "NOT_ASSESSED",
        }

        if value not in allowed:

            raise ValueError(
                "recommendation_assessment must be "
                "APPROPRIATE, INAPPROPRIATE, or NOT_ASSESSED."
            )

        return value

    # ------------------------------------------------------
    # Actual outcome validation
    # ------------------------------------------------------

    @field_validator(
        "actual_outcome"
    )
    @classmethod
    def validate_actual_outcome(
        cls,
        value: str,
    ) -> str:

        value = value.strip().upper()

        allowed = {
            "SEPSIS",
            "NO_SEPSIS",
            "UNKNOWN",
        }

        if value not in allowed:

            raise ValueError(
                "actual_outcome must be "
                "SEPSIS, NO_SEPSIS, or UNKNOWN."
            )

        return value


class PredictionFeedbackResponse(BaseModel):

    id: int

    prediction_id: int

    patient_id: int

    submitted_by: str

    prediction_assessment: str

    recommendation_assessment: str

    actual_outcome: str

    outcome_at: Optional[str] = None

    outcome_notes: Optional[str] = None

    validation_status: str

    validation_reason: str

    dataset_exported_at: Optional[str] = None

    created_at: str


class FeedbackStats(BaseModel):

    total: int

    eligible: int

    rejected: int

    pending: int

    unexported: int


# ─── Alert ────────────────────────────────────────────────────────────────────

class AlertResponse(BaseModel):

    id: int

    patient_id: int

    patient_name: str

    bed_number: str

    old_risk_level: Optional[str] = None

    new_risk_level: str

    probability: float

    triggered_at: str


# ─── Copilot ──────────────────────────────────────────────────────────────────

class CopilotResponse(BaseModel):

    patient_id: int

    question: str

    answer: str


# ─── Generic responses ────────────────────────────────────────────────────────

class SuccessResponse(BaseModel):

    success: bool = True

    message: str


class ErrorResponse(BaseModel):

    success: bool = False

    error: str

    detail: Optional[str] = None