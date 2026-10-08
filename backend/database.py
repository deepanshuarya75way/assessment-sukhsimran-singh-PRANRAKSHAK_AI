"""
Database layer for PranRakshak AI.

This module contains:
- SQLAlchemy engine/session configuration
- Patient and prediction models
- Prediction persistence/retrieval
- Prediction feedback persistence
- Feedback validation state management
- Training-dataset preparation helpers
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    String,
    Float,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, sessionmaker


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./pranrakshak.db",
)

# SQLite needs this option when using multiple threads,
# which is common with FastAPI/Uvicorn.
connect_args: dict[str, Any] = {}

is_sqlite = DATABASE_URL.startswith("sqlite")

if is_sqlite:
    connect_args["check_same_thread"] = False


engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.utcnow().isoformat()


def safe_json_loads(
    value: Optional[str],
    default: Any,
) -> Any:
    """Safely deserialize JSON stored in the database."""

    if not value:
        return default

    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------

class Patient(Base):
    __tablename__ = "patients"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    patient_id = Column(
        String,
        unique=True,
        nullable=False,
    )

    name = Column(
        String,
        nullable=False,
    )

    age = Column(
        Integer,
        nullable=False,
    )

    gender = Column(
        String,
        nullable=True,
    )

    admission_time = Column(
        String,
        nullable=True,
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    patient_id = Column(
        Integer,
        ForeignKey(
            "patients.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    sepsis_probability = Column(
        Float,
        nullable=False,
    )

    risk_level = Column(
        String,
        nullable=False,
    )

    shap_json = Column(
        String,
        nullable=False,
    )

    predicted_at = Column(
        String,
        nullable=False,
    )

    # ------------------------------------------------------------------
    # NEW: Feedback / training dataset fields
    # ------------------------------------------------------------------

    model_version = Column(
        String,
        nullable=False,
        default="1.0.0",
    )

    input_features_json = Column(
        String,
        nullable=False,
        default="{}",
    )

    clinical_recommendation = Column(
        String,
        nullable=False,
        default="",
    )


class PredictionFeedback(Base):
    """
    Clinician feedback associated with a specific prediction.

    One user can submit only one feedback record for a given prediction.
    """

    __tablename__ = "prediction_feedback"

    __table_args__ = (
        UniqueConstraint(
            "prediction_id",
            "submitted_by",
            name="uq_feedback_prediction_user",
        ),
    )

    id = Column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    prediction_id = Column(
        Integer,
        ForeignKey(
            "predictions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    patient_id = Column(
        Integer,
        ForeignKey(
            "patients.id",
            ondelete="CASCADE",
        ),
        nullable=False,
    )

    # Authorized clinician/user identifier
    submitted_by = Column(
        String,
        nullable=False,
    )

    # Whether the original prediction was correct
    prediction_assessment = Column(
        String,
        nullable=False,
    )

    # Whether the clinical escalation recommendation was appropriate
    recommendation_assessment = Column(
        String,
        nullable=False,
    )

    # Actual clinical outcome
    #
    # Supported values:
    #   SEPSIS
    #   NO_SEPSIS
    #   UNKNOWN
    actual_outcome = Column(
        String,
        nullable=False,
    )

    # When the actual outcome became known
    outcome_at = Column(
        String,
        nullable=True,
    )

    # Optional clinician notes
    outcome_notes = Column(
        String,
        nullable=True,
    )

    # Dataset preparation state
    #
    # PENDING
    # ELIGIBLE
    # REJECTED
    validation_status = Column(
        String,
        nullable=False,
    )

    validation_reason = Column(
        String,
        nullable=False,
    )

    # Prevent the same record from being exported repeatedly
    dataset_exported_at = Column(
        String,
        nullable=True,
    )

    created_at = Column(
        String,
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Database Initialization / Migration
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create missing tables and apply the small SQLite migration required
    for the feedback feature.

    SQLAlchemy's create_all() creates missing tables but does not modify
    existing tables by adding new columns. Therefore the three new
    Prediction columns are added explicitly for an existing SQLite DB.
    """

    Base.metadata.create_all(bind=engine)

    if is_sqlite:
        with engine.begin() as conn:

            # Find existing Prediction columns.
            result = conn.execute(
                text(
                    "PRAGMA table_info(predictions)"
                )
            )

            existing_columns = {
                row[1]
                for row in result
            }

            migrations = {
                "model_version": (
                    "ALTER TABLE predictions "
                    "ADD COLUMN model_version "
                    "TEXT NOT NULL DEFAULT '1.0.0'"
                ),

                "input_features_json": (
                    "ALTER TABLE predictions "
                    "ADD COLUMN input_features_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                ),

                "clinical_recommendation": (
                    "ALTER TABLE predictions "
                    "ADD COLUMN clinical_recommendation "
                    "TEXT NOT NULL DEFAULT ''"
                ),
            }

            for column_name, statement in migrations.items():

                if column_name not in existing_columns:

                    logger.info(
                        "Adding database column: predictions.%s",
                        column_name,
                    )

                    conn.execute(
                        text(statement)
                    )

    logger.info(
        "Database initialized successfully: %s",
        DATABASE_URL.split("@")[-1],
    )


# ---------------------------------------------------------------------------
# Patient Helpers
# ---------------------------------------------------------------------------

def get_patient_by_id(
    patient_id: int,
) -> Optional[dict[str, Any]]:
    """Return a patient by database ID."""

    db = SessionLocal()

    try:

        patient = (
            db.query(Patient)
            .filter(
                Patient.id == patient_id
            )
            .first()
        )

        if not patient:
            return None

        return {
            "id": patient.id,
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "admission_time": patient.admission_time,
        }

    finally:
        db.close()


def get_patient_by_external_id(
    patient_id: str,
) -> Optional[dict[str, Any]]:
    """Return a patient using the external patient identifier."""

    db = SessionLocal()

    try:

        patient = (
            db.query(Patient)
            .filter(
                Patient.patient_id == patient_id
            )
            .first()
        )

        if not patient:
            return None

        return {
            "id": patient.id,
            "patient_id": patient.patient_id,
            "name": patient.name,
            "age": patient.age,
            "gender": patient.gender,
            "admission_time": patient.admission_time,
        }

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Prediction Helpers
# ---------------------------------------------------------------------------

def save_prediction(
    patient_id: int,
    probability: float,
    risk_level: str,
    shap_factors: list[dict[str, Any]],
    input_features: Optional[dict[str, Any]] = None,
    model_version: str = "1.0.0",
    clinical_recommendation: str = "",
) -> int:
    """
    Save a model prediction together with the exact model input snapshot.

    The input feature snapshot is important for future training because
    feedback must be connected to the exact feature vector used when
    the original prediction was generated.
    """

    db = SessionLocal()

    try:

        prediction = Prediction(
            patient_id=patient_id,

            sepsis_probability=round(
                float(probability),
                4,
            ),

            risk_level=risk_level,

            shap_json=json.dumps(
                shap_factors,
                allow_nan=False,
            ),

            predicted_at=utc_now_iso(),

            model_version=model_version,

            input_features_json=json.dumps(
                input_features or {},
                allow_nan=False,
            ),

            clinical_recommendation=(
                clinical_recommendation
                or ""
            ),
        )

        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        return int(prediction.id)

    except Exception:
        db.rollback()
        logger.exception(
            "Failed to save prediction for patient %s",
            patient_id,
        )
        raise

    finally:
        db.close()


def get_prediction_by_id(
    prediction_id: int,
) -> Optional[dict[str, Any]]:
    """Return a complete prediction including model input snapshot."""

    db = SessionLocal()

    try:

        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.id == prediction_id
            )
            .first()
        )

        if not prediction:
            return None

        return {
            "id": prediction.id,

            "patient_id":
                prediction.patient_id,

            "sepsis_probability":
                prediction.sepsis_probability,

            "risk_level":
                prediction.risk_level,

            "shap_factors":
                safe_json_loads(
                    prediction.shap_json,
                    [],
                ),

            "predicted_at":
                prediction.predicted_at,

            "model_version":
                getattr(
                    prediction,
                    "model_version",
                    "1.0.0",
                ),

            "input_features":
                safe_json_loads(
                    getattr(
                        prediction,
                        "input_features_json",
                        "{}",
                    ),
                    {},
                ),

            "clinical_recommendation":
                getattr(
                    prediction,
                    "clinical_recommendation",
                    "",
                ),
        }

    finally:
        db.close()


def get_latest_prediction(
    patient_id: int,
) -> Optional[dict[str, Any]]:
    """Return the most recent prediction for a patient."""

    db = SessionLocal()

    try:

        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.patient_id == patient_id
            )
            .order_by(
                Prediction.id.desc()
            )
            .first()
        )

        if not prediction:
            return None

        return {
            "id": prediction.id,

            "patient_id":
                prediction.patient_id,

            "sepsis_probability":
                prediction.sepsis_probability,

            "risk_level":
                prediction.risk_level,

            "shap_factors":
                safe_json_loads(
                    prediction.shap_json,
                    [],
                ),

            "predicted_at":
                prediction.predicted_at,

            "model_version":
                getattr(
                    prediction,
                    "model_version",
                    "1.0.0",
                ),

            "input_features":
                safe_json_loads(
                    getattr(
                        prediction,
                        "input_features_json",
                        "{}",
                    ),
                    {},
                ),

            "clinical_recommendation":
                getattr(
                    prediction,
                    "clinical_recommendation",
                    "",
                ),
        }

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Feedback Helpers
# ---------------------------------------------------------------------------

def feedback_to_dict(
    row: PredictionFeedback,
) -> dict[str, Any]:
    """Convert PredictionFeedback ORM object into a JSON-safe dictionary."""

    return {
        "id": row.id,

        "prediction_id":
            row.prediction_id,

        "patient_id":
            row.patient_id,

        "submitted_by":
            row.submitted_by,

        "prediction_assessment":
            row.prediction_assessment,

        "recommendation_assessment":
            row.recommendation_assessment,

        "actual_outcome":
            row.actual_outcome,

        "outcome_at":
            row.outcome_at,

        "outcome_notes":
            row.outcome_notes,

        "validation_status":
            row.validation_status,

        "validation_reason":
            row.validation_reason,

        "dataset_exported_at":
            row.dataset_exported_at,

        "created_at":
            row.created_at,
    }


def create_feedback(
    prediction_id: int,
    patient_id: int,
    submitted_by: str,
    prediction_assessment: str,
    recommendation_assessment: str,
    actual_outcome: str,
    outcome_at: Optional[str],
    outcome_notes: Optional[str],
    validation_status: str,
    validation_reason: str,
) -> dict[str, Any]:
    """Create a clinician feedback record."""

    db = SessionLocal()

    try:

        feedback = PredictionFeedback(
            prediction_id=prediction_id,

            patient_id=patient_id,

            submitted_by=submitted_by,

            prediction_assessment=(
                prediction_assessment
            ),

            recommendation_assessment=(
                recommendation_assessment
            ),

            actual_outcome=actual_outcome,

            outcome_at=outcome_at,

            outcome_notes=outcome_notes,

            validation_status=validation_status,

            validation_reason=validation_reason,

            created_at=utc_now_iso(),
        )

        db.add(feedback)
        db.commit()
        db.refresh(feedback)

        return feedback_to_dict(
            feedback
        )

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


def get_feedback(
    feedback_id: int,
) -> Optional[dict[str, Any]]:
    """Return feedback by ID."""

    db = SessionLocal()

    try:

        row = (
            db.query(PredictionFeedback)
            .filter(
                PredictionFeedback.id
                == feedback_id
            )
            .first()
        )

        if not row:
            return None

        return feedback_to_dict(row)

    finally:
        db.close()


def get_feedback_by_prediction_and_user(
    prediction_id: int,
    submitted_by: str,
) -> Optional[dict[str, Any]]:
    """Check whether a user has already submitted feedback."""

    db = SessionLocal()

    try:

        row = (
            db.query(PredictionFeedback)
            .filter(
                PredictionFeedback.prediction_id
                == prediction_id,

                PredictionFeedback.submitted_by
                == submitted_by,
            )
            .first()
        )

        if not row:
            return None

        return feedback_to_dict(row)

    finally:
        db.close()


def update_feedback(
    feedback_id: int,
    prediction_assessment: str,
    recommendation_assessment: str,
    actual_outcome: str,
    outcome_at: Optional[str],
    outcome_notes: Optional[str],
    validation_status: str,
    validation_reason: str,
) -> Optional[dict[str, Any]]:
    """
    Update feedback.

    Used primarily when an initially UNKNOWN clinical outcome
    becomes available later.
    """

    db = SessionLocal()

    try:

        row = (
            db.query(PredictionFeedback)
            .filter(
                PredictionFeedback.id
                == feedback_id
            )
            .first()
        )

        if not row:
            return None

        row.prediction_assessment = (
            prediction_assessment
        )

        row.recommendation_assessment = (
            recommendation_assessment
        )

        row.actual_outcome = (
            actual_outcome
        )

        row.outcome_at = outcome_at

        row.outcome_notes = outcome_notes

        row.validation_status = (
            validation_status
        )

        row.validation_reason = (
            validation_reason
        )

        # If the record changes after being rejected/pending,
        # it should be eligible for a fresh export.
        if validation_status != "ELIGIBLE":
            row.dataset_exported_at = None

        db.commit()
        db.refresh(row)

        return feedback_to_dict(row)

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Feedback Statistics
# ---------------------------------------------------------------------------

def get_feedback_stats() -> dict[str, int]:
    """Return counts of feedback records by preparation state."""

    db = SessionLocal()

    try:

        total = (
            db.query(
                PredictionFeedback
            ).count()
        )

        eligible = (
            db.query(
                PredictionFeedback
            )
            .filter(
                PredictionFeedback.validation_status
                == "ELIGIBLE"
            )
            .count()
        )

        rejected = (
            db.query(
                PredictionFeedback
            )
            .filter(
                PredictionFeedback.validation_status
                == "REJECTED"
            )
            .count()
        )

        pending = (
            db.query(
                PredictionFeedback
            )
            .filter(
                PredictionFeedback.validation_status
                == "PENDING"
            )
            .count()
        )

        unexported = (
            db.query(
                PredictionFeedback
            )
            .filter(
                PredictionFeedback.validation_status
                == "ELIGIBLE",

                PredictionFeedback.dataset_exported_at
                .is_(None),
            )
            .count()
        )

        return {
            "total": total,
            "eligible": eligible,
            "rejected": rejected,
            "pending": pending,
            "unexported": unexported,
        }

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Training Dataset Preparation
# ---------------------------------------------------------------------------

def get_eligible_feedback_for_dataset(
    include_exported: bool = False,
) -> list[dict[str, Any]]:
    """
    Return validated feedback joined with its original prediction.

    Only ELIGIBLE records are returned.

    By default, already-exported records are excluded to prevent
    duplicate dataset rows.
    """

    db = SessionLocal()

    try:

        query = (
            db.query(
                PredictionFeedback,
                Prediction,
            )
            .join(
                Prediction,
                Prediction.id
                == PredictionFeedback.prediction_id,
            )
            .filter(
                PredictionFeedback.validation_status
                == "ELIGIBLE",
            )
        )

        if not include_exported:

            query = query.filter(
                PredictionFeedback.dataset_exported_at
                .is_(None)
            )

        rows = (
            query
            .order_by(
                PredictionFeedback.id.asc()
            )
            .all()
        )

        result: list[dict[str, Any]] = []

        for feedback, prediction in rows:

            result.append(
                {
                    "feedback":
                        feedback_to_dict(
                            feedback
                        ),

                    "prediction": {
                        "id":
                            prediction.id,

                        "patient_id":
                            prediction.patient_id,

                        "sepsis_probability":
                            prediction.sepsis_probability,

                        "risk_level":
                            prediction.risk_level,

                        "predicted_at":
                            prediction.predicted_at,

                        "model_version":
                            getattr(
                                prediction,
                                "model_version",
                                "1.0.0",
                            ),

                        "input_features":
                            safe_json_loads(
                                getattr(
                                    prediction,
                                    "input_features_json",
                                    "{}",
                                ),
                                {},
                            ),
                    },
                }
            )

        return result

    finally:
        db.close()


def mark_feedback_exported(
    feedback_ids: list[int],
) -> None:
    """
    Mark feedback records as exported.

    This prevents the same eligible records from appearing in
    subsequent incremental dataset exports.
    """

    if not feedback_ids:
        return

    db = SessionLocal()

    try:

        now = utc_now_iso()

        (
            db.query(
                PredictionFeedback
            )
            .filter(
                PredictionFeedback.id.in_(
                    feedback_ids
                )
            )
            .update(
                {
                    PredictionFeedback.dataset_exported_at:
                        now,
                },
                synchronize_session=False,
            )
        )

        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Optional Dependency
# ---------------------------------------------------------------------------

def get_db():
    """
    FastAPI dependency for routes that need a database session.
    """

    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()