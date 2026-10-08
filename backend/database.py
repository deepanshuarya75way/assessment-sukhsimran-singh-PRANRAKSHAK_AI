"""
Database layer for PranRakshak AI.

Provides:
- SQLAlchemy engine/session configuration
- Patient persistence
- Patient vitals persistence
- Prediction persistence/retrieval
- Prediction feedback persistence
- Feedback validation state management
- Training-dataset preparation helpers

The ORM uses SQLAlchemy 2.x typed declarative mappings so Pylance can
understand mapped attributes correctly.
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timezone
from typing import Any, Generator, Optional
from uuid import uuid4

from sqlalchemy import (
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./pranrakshak.db",
)

is_sqlite = DATABASE_URL.startswith("sqlite")

connect_args: dict[str, Any] = {}

if is_sqlite:
    connect_args["check_same_thread"] = False


engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


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


def _json_dumps(
    value: Any,
    default: Any,
) -> str:
    """Serialize JSON safely, falling back to a default value."""

    try:
        return json.dumps(
            value if value is not None else default,
            allow_nan=False,
        )
    except (TypeError, ValueError):
        return json.dumps(
            default,
            allow_nan=False,
        )


def _finite_float(
    value: Any,
) -> Optional[float]:
    """Convert a value to a finite float, otherwise return None."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    return number if math.isfinite(number) else None


def _safe_int(
    value: Any,
) -> Optional[int]:
    """Convert a value to int, otherwise return None."""

    try:
        if value is None or value == "":
            return None

        return int(float(value))

    except (TypeError, ValueError):
        return None


def _patient_to_dict(
    patient: "Patient",
) -> dict[str, Any]:
    """Convert a Patient ORM object to the shape expected by API routes."""

    return {
        "id": patient.id,
        "patient_id": patient.patient_id,
        "name": patient.name,
        "bed_number": patient.bed_number,
        "age": patient.age,
        "gender": patient.gender,
        "created_at": patient.created_at,

        # Legacy compatibility.
        "admission_time": patient.admission_time,
    }


def _prediction_to_dict(
    prediction: "Prediction",
) -> dict[str, Any]:
    """Convert a Prediction ORM object to a JSON-safe dictionary."""

    return {
        "id": prediction.id,
        "patient_id": prediction.patient_id,
        "sepsis_probability": prediction.sepsis_probability,
        "risk_level": prediction.risk_level,
        "shap_factors": safe_json_loads(
            prediction.shap_json,
            [],
        ),
        "predicted_at": prediction.predicted_at,
        "model_version": prediction.model_version,
        "input_features": safe_json_loads(
            prediction.input_features_json,
            {},
        ),
        "clinical_recommendation": (
            prediction.clinical_recommendation
        ),
    }


# ---------------------------------------------------------------------------
# Database Models
# ---------------------------------------------------------------------------

class Patient(Base):
    __tablename__ = "patients"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # External identifier retained for compatibility.
    patient_id: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
    )

    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    bed_number: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
    )

    age: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    gender: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
    )

    created_at: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    # Legacy field retained for existing databases.
    admission_time: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )


class PatientVital(Base):
    __tablename__ = "patient_vitals"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "patients.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    hour: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    hr: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    o2sat: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    temp: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    sbp: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    map_val: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    resp: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    wbc: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    creatinine: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    glucose: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
    )

    age: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    iculos: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )

    row_index: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "patients.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    sepsis_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    risk_level: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    shap_json: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    predicted_at: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="1.0.0",
    )

    input_features_json: Mapped[str] = mapped_column(
        String,
        nullable=False,
        default="{}",
    )

    clinical_recommendation: Mapped[str] = mapped_column(
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

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    prediction_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "predictions.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    patient_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey(
            "patients.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    submitted_by: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    prediction_assessment: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    recommendation_assessment: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    actual_outcome: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    outcome_at: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    outcome_notes: Mapped[Optional[str]] = mapped_column(
        String,
        nullable=True,
    )

    validation_status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    validation_reason: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    dataset_exported_at: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
    )

    created_at: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# Database Initialization / Migration
# ---------------------------------------------------------------------------

def _sqlite_add_column_if_missing(
    conn: Any,
    table_name: str,
    column_name: str,
    ddl: str,
) -> None:
    """Add a SQLite column only when it is not already present."""

    columns = {
        row[1]
        for row in conn.execute(
            text(
                f"PRAGMA table_info({table_name})"
            )
        )
    }

    if column_name not in columns:
        logger.info(
            "Adding database column: %s.%s",
            table_name,
            column_name,
        )

        conn.execute(
            text(ddl)
        )


def _postgres_add_column_if_missing(
    conn: Any,
    table_name: str,
    column_name: str,
    ddl: str,
) -> None:
    """Add a PostgreSQL column only when it is not already present."""

    conn.execute(
        text(
            f"ALTER TABLE {table_name} "
            f"ADD COLUMN IF NOT EXISTS "
            f"{column_name} {ddl}"
        )
    )


def init_db() -> None:
    """Create tables and migrate columns required by the current API."""

    Base.metadata.create_all(
        bind=engine
    )

    dialect = engine.dialect.name

    with engine.begin() as conn:

        if dialect == "sqlite":

            # -----------------------------------------------------------
            # Existing patients table migration
            # -----------------------------------------------------------

            _sqlite_add_column_if_missing(
                conn,
                "patients",
                "bed_number",
                (
                    "ALTER TABLE patients "
                    "ADD COLUMN bed_number TEXT"
                ),
            )

            _sqlite_add_column_if_missing(
                conn,
                "patients",
                "created_at",
                (
                    "ALTER TABLE patients "
                    "ADD COLUMN created_at TEXT"
                ),
            )

            # -----------------------------------------------------------
            # Existing predictions table migration
            # -----------------------------------------------------------

            _sqlite_add_column_if_missing(
                conn,
                "predictions",
                "model_version",
                (
                    "ALTER TABLE predictions "
                    "ADD COLUMN model_version "
                    "TEXT NOT NULL DEFAULT '1.0.0'"
                ),
            )

            _sqlite_add_column_if_missing(
                conn,
                "predictions",
                "input_features_json",
                (
                    "ALTER TABLE predictions "
                    "ADD COLUMN input_features_json "
                    "TEXT NOT NULL DEFAULT '{}'"
                ),
            )

            _sqlite_add_column_if_missing(
                conn,
                "predictions",
                "clinical_recommendation",
                (
                    "ALTER TABLE predictions "
                    "ADD COLUMN clinical_recommendation "
                    "TEXT NOT NULL DEFAULT ''"
                ),
            )

            # Backfill existing patient timestamps.
            conn.execute(
                text(
                    "UPDATE patients "
                    "SET created_at = "
                    "COALESCE(created_at, admission_time, :now) "
                    "WHERE created_at IS NULL "
                    "OR created_at = ''"
                ),
                {
                    "now": utc_now_iso()
                },
            )

        elif dialect == "postgresql":

            # -----------------------------------------------------------
            # Existing patients table migration
            # -----------------------------------------------------------

            _postgres_add_column_if_missing(
                conn,
                "patients",
                "bed_number",
                "TEXT",
            )

            _postgres_add_column_if_missing(
                conn,
                "patients",
                "created_at",
                "TEXT",
            )

            # -----------------------------------------------------------
            # Existing predictions table migration
            # -----------------------------------------------------------

            _postgres_add_column_if_missing(
                conn,
                "predictions",
                "model_version",
                "TEXT NOT NULL DEFAULT '1.0.0'",
            )

            _postgres_add_column_if_missing(
                conn,
                "predictions",
                "input_features_json",
                "TEXT NOT NULL DEFAULT '{}'",
            )

            _postgres_add_column_if_missing(
                conn,
                "predictions",
                "clinical_recommendation",
                "TEXT NOT NULL DEFAULT ''",
            )

            conn.execute(
                text(
                    "UPDATE patients "
                    "SET created_at = "
                    "COALESCE(created_at, admission_time, :now) "
                    "WHERE created_at IS NULL "
                    "OR created_at = ''"
                ),
                {
                    "now": utc_now_iso()
                },
            )

    if is_sqlite:

        with engine.begin() as conn:
            conn.execute(
                text(
                    "PRAGMA foreign_keys=ON"
                )
            )

    logger.info(
        "Database initialized successfully: %s",
        DATABASE_URL.split("@")[-1],
    )


# ---------------------------------------------------------------------------
# Patient Helpers
# ---------------------------------------------------------------------------

def create_patient(
    name: str,
    bed_number: Optional[str],
    age: int,
    gender: Optional[str],
) -> dict[str, Any]:
    """Create a patient and return the API-compatible patient dictionary."""

    db = SessionLocal()

    try:

        now = utc_now_iso()

        external_id = (
            f"PR-{uuid4().hex[:10].upper()}"
        )

        patient = Patient(
            patient_id=external_id,
            name=name.strip(),
            bed_number=(
                bed_number.strip()
                if bed_number
                else None
            ),
            age=int(age),
            gender=gender,
            created_at=now,
            admission_time=now,
        )

        db.add(patient)
        db.commit()
        db.refresh(patient)

        return _patient_to_dict(
            patient
        )

    except Exception:
        db.rollback()

        logger.exception(
            "Failed to create patient"
        )

        raise

    finally:
        db.close()


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

        return _patient_to_dict(
            patient
        )

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

        return _patient_to_dict(
            patient
        )

    finally:
        db.close()


def get_all_patients() -> list[dict[str, Any]]:
    """Return all patients with their latest prediction summary."""

    db = SessionLocal()

    try:

        patients = (
            db.query(Patient)
            .order_by(
                Patient.id.asc()
            )
            .all()
        )

        result: list[dict[str, Any]] = []

        for patient in patients:

            latest = (
                db.query(Prediction)
                .filter(
                    Prediction.patient_id
                    == patient.id
                )
                .order_by(
                    Prediction.id.desc()
                )
                .first()
            )

            item = _patient_to_dict(
                patient
            )

            item.update(
                {
                    "latest_probability": (
                        latest.sepsis_probability
                        if latest
                        else None
                    ),

                    "latest_risk_level": (
                        latest.risk_level
                        if latest
                        else None
                    ),

                    "latest_predicted_at": (
                        latest.predicted_at
                        if latest
                        else None
                    ),

                    "latest_prediction_id": (
                        latest.id
                        if latest
                        else None
                    ),

                    "model_version": (
                        latest.model_version
                        if latest
                        else None
                    ),
                }
            )

            result.append(item)

        return result

    finally:
        db.close()


def delete_patient(
    patient_id: int,
) -> bool:
    """Delete a patient and all related records."""

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
            return False

        # Explicit deletion keeps this reliable even for older SQLite DBs.
        db.query(
            PredictionFeedback
        ).filter(
            PredictionFeedback.patient_id
            == patient_id
        ).delete(
            synchronize_session=False
        )

        db.query(
            Prediction
        ).filter(
            Prediction.patient_id
            == patient_id
        ).delete(
            synchronize_session=False
        )

        db.query(
            PatientVital
        ).filter(
            PatientVital.patient_id
            == patient_id
        ).delete(
            synchronize_session=False
        )

        db.delete(patient)

        db.commit()

        return True

    except Exception:
        db.rollback()

        logger.exception(
            "Failed to delete patient %s",
            patient_id,
        )

        raise

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Vitals Helpers
# ---------------------------------------------------------------------------

def _get_value(
    row: Any,
    *names: str,
) -> Any:
    """Get the first available value from a pandas-like row."""

    for name in names:

        try:
            value = row[name]

        except (
            KeyError,
            TypeError,
            IndexError,
        ):
            continue

        # Handle pandas NaN.
        if (
            isinstance(value, float)
            and math.isnan(value)
        ):
            return None

        return value

    return None


def save_vitals_batch(
    patient_id: int,
    df: Any,
) -> int:
    """Persist a batch of patient vitals from a pandas-like DataFrame."""

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
            raise ValueError(
                f"Patient {patient_id} does not exist"
            )

        # Replace the current uploaded vitals for this patient.
        db.query(
            PatientVital
        ).filter(
            PatientVital.patient_id
            == patient_id
        ).delete(
            synchronize_session=False
        )

        if hasattr(
            df,
            "to_dict",
        ):
            records = df.to_dict(
                orient="records"
            )
        else:
            records = list(df)

        objects: list[PatientVital] = []

        aliases = {
            "hour": (
                "hour",
                "Hour",
                "hours",
                "Hours",
            ),

            "hr": (
                "hr",
                "HR",
                "heart_rate",
                "HeartRate",
            ),

            "o2sat": (
                "o2sat",
                "O2Sat",
                "O2SAT",
                "spo2",
                "SpO2",
            ),

            "temp": (
                "temp",
                "Temp",
                "temperature",
                "Temperature",
            ),

            "sbp": (
                "sbp",
                "SBP",
                "systolic_bp",
                "SystolicBP",
            ),

            "map_val": (
                "map_val",
                "MAP",
                "map",
                "mean_arterial_pressure",
            ),

            "resp": (
                "resp",
                "Resp",
                "respiratory_rate",
                "RespRate",
            ),

            "wbc": (
                "wbc",
                "WBC",
                "white_blood_cell_count",
            ),

            "creatinine": (
                "creatinine",
                "Creatinine",
            ),

            "glucose": (
                "glucose",
                "Glucose",
            ),

            "age": (
                "age",
                "Age",
            ),

            "iculos": (
                "iculos",
                "ICULOS",
                "iculos_hours",
                "ICU_LOS",
            ),

            "row_index": (
                "row_index",
                "row_idx",
                "index",
            ),
        }

        for index, row in enumerate(records):

            row_index = _safe_int(
                _get_value(
                    row,
                    *aliases["row_index"],
                )
            )

            objects.append(
                PatientVital(
                    patient_id=patient_id,

                    hour=_finite_float(
                        _get_value(
                            row,
                            *aliases["hour"],
                        )
                    ),

                    hr=_finite_float(
                        _get_value(
                            row,
                            *aliases["hr"],
                        )
                    ),

                    o2sat=_finite_float(
                        _get_value(
                            row,
                            *aliases["o2sat"],
                        )
                    ),

                    temp=_finite_float(
                        _get_value(
                            row,
                            *aliases["temp"],
                        )
                    ),

                    sbp=_finite_float(
                        _get_value(
                            row,
                            *aliases["sbp"],
                        )
                    ),

                    map_val=_finite_float(
                        _get_value(
                            row,
                            *aliases["map_val"],
                        )
                    ),

                    resp=_finite_float(
                        _get_value(
                            row,
                            *aliases["resp"],
                        )
                    ),

                    wbc=_finite_float(
                        _get_value(
                            row,
                            *aliases["wbc"],
                        )
                    ),

                    creatinine=_finite_float(
                        _get_value(
                            row,
                            *aliases["creatinine"],
                        )
                    ),

                    glucose=_finite_float(
                        _get_value(
                            row,
                            *aliases["glucose"],
                        )
                    ),

                    age=_safe_int(
                        _get_value(
                            row,
                            *aliases["age"],
                        )
                    ),

                    iculos=_safe_int(
                        _get_value(
                            row,
                            *aliases["iculos"],
                        )
                    ),

                    row_index=(
                        row_index
                        if row_index is not None
                        else index
                    ),
                )
            )

        if objects:
            db.add_all(objects)

        db.commit()

        return len(objects)

    except Exception:
        db.rollback()

        logger.exception(
            "Failed to save vitals for patient %s",
            patient_id,
        )

        raise

    finally:
        db.close()


def get_vitals_for_patient(
    patient_id: int,
) -> list[dict[str, Any]]:
    """Return stored vitals in the shape expected by the API models."""

    db = SessionLocal()

    try:

        rows = (
            db.query(PatientVital)
            .filter(
                PatientVital.patient_id
                == patient_id
            )
            .order_by(
                PatientVital.row_index.asc(),
                PatientVital.id.asc(),
            )
            .all()
        )

        return [
            {
                "hour": row.hour,
                "hr": row.hr,
                "o2sat": row.o2sat,
                "temp": row.temp,
                "sbp": row.sbp,
                "map_val": row.map_val,
                "resp": row.resp,
                "wbc": row.wbc,
                "creatinine": row.creatinine,
                "glucose": row.glucose,
                "age": row.age,
                "iculos": row.iculos,
                "row_index": row.row_index,
            }
            for row in rows
        ]

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

            shap_json=_json_dumps(
                shap_factors,
                [],
            ),

            predicted_at=utc_now_iso(),

            model_version=(
                model_version
                or "1.0.0"
            ),

            input_features_json=_json_dumps(
                input_features,
                {},
            ),

            clinical_recommendation=(
                clinical_recommendation
                or ""
            ),
        )

        db.add(prediction)

        db.commit()

        db.refresh(prediction)

        return int(
            prediction.id
        )

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
    """Return a complete prediction including its input snapshot."""

    db = SessionLocal()

    try:

        prediction = (
            db.query(Prediction)
            .filter(
                Prediction.id
                == prediction_id
            )
            .first()
        )

        if not prediction:
            return None

        return _prediction_to_dict(
            prediction
        )

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
                Prediction.patient_id
                == patient_id
            )
            .order_by(
                Prediction.id.desc()
            )
            .first()
        )

        if not prediction:
            return None

        return _prediction_to_dict(
            prediction
        )

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
        "prediction_id": row.prediction_id,
        "patient_id": row.patient_id,
        "submitted_by": row.submitted_by,
        "prediction_assessment": (
            row.prediction_assessment
        ),
        "recommendation_assessment": (
            row.recommendation_assessment
        ),
        "actual_outcome": row.actual_outcome,
        "outcome_at": row.outcome_at,
        "outcome_notes": row.outcome_notes,
        "validation_status": (
            row.validation_status
        ),
        "validation_reason": (
            row.validation_reason
        ),
        "dataset_exported_at": (
            row.dataset_exported_at
        ),
        "created_at": row.created_at,
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
            validation_status=(
                validation_status
            ),
            validation_reason=(
                validation_reason
            ),
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

        return feedback_to_dict(
            row
        )

    finally:
        db.close()


def get_feedback_by_prediction_and_user(
    prediction_id: int,
    submitted_by: str,
) -> Optional[dict[str, Any]]:
    """Return existing feedback for a prediction/user pair, if present."""

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

        return feedback_to_dict(
            row
        )

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
    """Update an existing feedback record."""

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

        row.outcome_notes = (
            outcome_notes
        )

        row.validation_status = (
            validation_status
        )

        row.validation_reason = (
            validation_reason
        )

        # Any update invalidates a previous export marker.
        row.dataset_exported_at = None

        db.commit()

        db.refresh(row)

        return feedback_to_dict(
            row
        )

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
    duplicate incremental dataset rows.
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
                    "feedback": feedback_to_dict(
                        feedback
                    ),

                    "prediction": {
                        "id": prediction.id,

                        "patient_id": (
                            prediction.patient_id
                        ),

                        "sepsis_probability": (
                            prediction.sepsis_probability
                        ),

                        "risk_level": (
                            prediction.risk_level
                        ),

                        "predicted_at": (
                            prediction.predicted_at
                        ),

                        "model_version": (
                            prediction.model_version
                        ),

                        "input_features": (
                            safe_json_loads(
                                prediction.input_features_json,
                                {},
                            )
                        ),

                        "shap_factors": (
                            safe_json_loads(
                                prediction.shap_json,
                                [],
                            )
                        ),

                        "clinical_recommendation": (
                            prediction.clinical_recommendation
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
    """Mark eligible feedback records as exported."""

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
                ),

                PredictionFeedback.validation_status
                == "ELIGIBLE",
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
# Optional FastAPI Dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session."""

    db = SessionLocal()

    try:
        yield db

    finally:
        db.close()