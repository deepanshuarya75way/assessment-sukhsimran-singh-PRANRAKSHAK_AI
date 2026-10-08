import joblib
import pandas as pd
import numpy as np

from pathlib import Path
from fastapi import HTTPException
from typing import Optional

from config import get_settings
from logging_config import get_logger
from utils import REQUIRED_FEATURES

logger = get_logger(__name__)
settings = get_settings()

_pipeline = None


# ----------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------

def load_model():
    """
    Load the trained sklearn Pipeline once at startup.
    """

    global _pipeline

    path = Path(settings.model_path)

    if not path.exists():
        logger.error(
            "Model file not found at %s",
            path
        )

        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    _pipeline = joblib.load(path)

    logger.info(
        "Model loaded from %s",
        path
    )

    # Debug once — helps confirm schema
    try:
        logger.info(
            "Model expects features: %s",
            list(_pipeline.feature_names_in_)
        )
    except Exception:
        pass

    return _pipeline


# ----------------------------------------------------------
# RISK LOGIC
# ----------------------------------------------------------

def _probability_to_risk(
    probability: float
) -> str:
    """
    Convert model probability into dashboard risk category.

    Thresholds are based on the existing project logic.
    """

    if probability >= 0.65:
        return "HIGH"

    elif probability >= 0.35:
        return "MEDIUM"

    return "LOW"


# ----------------------------------------------------------
# CLINICAL RECOMMENDATION
# ----------------------------------------------------------

def clinical_recommendation(
    risk_level: str
) -> str:
    """
    Generate a non-prescriptive escalation recommendation
    based on the model risk category.

    This is decision support only and does not replace
    clinical judgment.
    """

    recommendations = {
        "HIGH": (
            "Immediate clinical review recommended. "
            "Consider urgent sepsis assessment and "
            "appropriate escalation according to hospital protocol."
        ),

        "MEDIUM": (
            "Close clinical monitoring recommended. "
            "Reassess vital signs and consider further "
            "sepsis evaluation according to clinical judgment."
        ),

        "LOW": (
            "Continue routine monitoring. "
            "Reassess if the patient's clinical condition changes."
        ),
    }

    return recommendations.get(
        risk_level,
        "Clinical review recommended."
    )


# ----------------------------------------------------------
# FEATURE ENGINEERING
# ----------------------------------------------------------

BASE_FEATURES = [
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "Resp",
    "WBC",
    "Creatinine",
    "Glucose",
]


def _engineer_features(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert raw time-series vitals into model-required features.

    Generates:

        HR_last
        HR_mean
        HR_std
        HR_trend

    and equivalent features for the remaining vital/lab columns.

    Also generates:

        Age
        ICULOS
    """

    features = {}

    for col in BASE_FEATURES:

        if col not in df.columns:
            continue

        series = df[col].dropna()

        if len(series) == 0:
            continue

        # --------------------------------------------------
        # Last value
        # --------------------------------------------------

        features[
            f"{col}_last"
        ] = float(
            series.iloc[-1]
        )

        # --------------------------------------------------
        # Mean
        # --------------------------------------------------

        features[
            f"{col}_mean"
        ] = float(
            series.mean()
        )

        # --------------------------------------------------
        # Standard deviation
        # --------------------------------------------------

        if len(series) > 1:

            std_value = float(
                series.std()
            )

            # A single observation has NaN std.
            # Store 0 instead so the feedback snapshot
            # always contains a valid numeric value.
            if not np.isfinite(std_value):
                std_value = 0.0

        else:
            std_value = 0.0

        features[
            f"{col}_std"
        ] = std_value

        # --------------------------------------------------
        # Trend / slope
        # --------------------------------------------------

        if len(series) > 1:

            x = np.arange(
                len(series)
            )

            slope = np.polyfit(
                x,
                series,
                1
            )[0]

        else:

            slope = 0.0

        features[
            f"{col}_trend"
        ] = float(slope)

    # ------------------------------------------------------
    # Required static features
    # ------------------------------------------------------

    if "Age" in df.columns:

        age_value = df["Age"].iloc[-1]

        if not pd.isna(age_value):

            features["Age"] = float(
                age_value
            )

    if "ICULOS" in df.columns:

        iculos_value = df["ICULOS"].iloc[-1]

        if not pd.isna(iculos_value):

            features["ICULOS"] = float(
                iculos_value
            )

    features_df = pd.DataFrame(
        [features]
    )

    return features_df


# ----------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------

def predict(
    df: pd.DataFrame
) -> dict:
    """
    Run sepsis prediction on validated patient time-series.

    Returns:

        {
            "probability": float,
            "risk_level": str,
            "features": {
                ...
            }
        }

    The `features` dictionary contains the exact engineered
    model input snapshot needed by the feedback/training
    dataset pipeline.
    """

    if _pipeline is None:

        raise HTTPException(
            status_code=503,
            detail="Prediction model is not loaded."
        )

    try:

        # --------------------------------------------------
        # Step 1 — Engineer features
        # --------------------------------------------------

        features_df = _engineer_features(
            df
        )

        logger.debug(
            "Engineered features columns: %s",
            list(features_df.columns)
        )

        # --------------------------------------------------
        # Step 2 — Validate required model features
        # --------------------------------------------------

        missing_features = [
            feature
            for feature in REQUIRED_FEATURES
            if feature not in features_df.columns
        ]

        if missing_features:

            logger.error(
                "Missing required model features: %s",
                missing_features
            )

            raise HTTPException(
                status_code=422,
                detail={
                    "message":
                        "Required model features are missing.",
                    "missing_features":
                        missing_features,
                },
            )

        # Keep exactly the feature order expected by the model.
        try:

            model_features = list(
                _pipeline.feature_names_in_
            )

            features_df = features_df[
                model_features
            ]

        except Exception:

            # If feature_names_in_ isn't available,
            # preserve the existing dataframe.
            pass

        # --------------------------------------------------
        # Step 3 — Validate numerical values
        # --------------------------------------------------

        if features_df.isnull().any().any():

            logger.error(
                "Engineered feature vector contains NaN values."
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    "Unable to generate a complete "
                    "model input feature vector."
                ),
            )

        if not np.isfinite(
            features_df.to_numpy(
                dtype=float
            )
        ).all():

            logger.error(
                "Engineered feature vector contains "
                "non-finite values."
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    "Model input contains non-finite values."
                ),
            )

        # --------------------------------------------------
        # Step 4 — Model prediction
        # --------------------------------------------------

        prob_array = (
            _pipeline.predict_proba(
                features_df
            )
        )

        probability = float(
            prob_array[0][1]
        )

        # --------------------------------------------------
        # Hybrid rule-based override
        # --------------------------------------------------

        def _rule_based_override(
            raw_df: pd.DataFrame
        ) -> Optional[str]:

            try:

                if (
                    raw_df is None
                    or raw_df.empty
                ):
                    return None

                last = raw_df.iloc[-1]

                # Safely extract values.
                def _val(col):

                    value = (
                        last.get(col)
                        if col in last.index
                        else None
                    )

                    if value is None:
                        return None

                    if pd.isna(value):
                        return None

                    return float(value)

                hr = _val("HR")
                o2 = _val("O2Sat")
                sbp = _val("SBP")
                temp = _val("Temp")

                # --------------------------------------------------
                # Critical overrides → HIGH
                # --------------------------------------------------

                if (
                    hr is not None
                    and hr > 180
                ) or (
                    o2 is not None
                    and o2 < 70
                ) or (
                    sbp is not None
                    and sbp < 70
                ) or (
                    temp is not None
                    and temp > 41
                ):

                    logger.info(
                        "Rule override: detected critical vitals "
                        "-> HIGH (hr=%s o2=%s sbp=%s temp=%s)",
                        hr,
                        o2,
                        sbp,
                        temp,
                    )

                    return "HIGH"

                # --------------------------------------------------
                # Healthy override → LOW
                # --------------------------------------------------

                healthy_checks = [
                    (
                        hr is not None
                        and hr < 80
                    ),

                    (
                        o2 is not None
                        and o2 > 98
                    ),

                    (
                        sbp is not None
                        and sbp > 110
                    ),

                    (
                        temp is not None
                        and temp < 37.2
                    ),
                ]

                if all(healthy_checks):

                    logger.info(
                        "Rule override: detected healthy vitals "
                        "-> LOW (hr=%s o2=%s sbp=%s temp=%s)",
                        hr,
                        o2,
                        sbp,
                        temp,
                    )

                    return "LOW"

            except Exception as exc:

                logger.debug(
                    "Rule override evaluation failed: %s",
                    exc,
                    exc_info=True,
                )

            return None

        override = _rule_based_override(
            df
        )

        # --------------------------------------------------
        # Apply override
        # --------------------------------------------------

        if override == "HIGH":

            probability = 0.92
            risk_level = "HIGH"

            logger.info(
                "Rule override applied: "
                "risk=HIGH probability=0.92"
            )

        elif override == "LOW":

            probability = 0.08
            risk_level = "LOW"

            logger.info(
                "Rule override applied: "
                "risk=LOW probability=0.08"
            )

        else:

            risk_level = _probability_to_risk(
                probability
            )

            logger.info(
                "ML thresholding applied: "
                "prob=%.4f risk=%s",
                probability,
                risk_level,
            )

    except HTTPException:
        raise

    except Exception as exc:

        logger.error(
            "Prediction failed: %s",
            exc,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction pipeline error: "
                f"{str(exc)}"
            ),
        )

    logger.info(
        "Prediction complete: "
        "prob=%.4f risk=%s rows_in_csv=%d",
        probability,
        risk_level,
        len(df),
    )

    # ------------------------------------------------------
    # Convert engineered features to JSON-safe values
    # ------------------------------------------------------

    feature_snapshot = {}

    for key, value in (
        features_df.iloc[0]
        .to_dict()
        .items()
    ):

        numeric_value = float(value)

        if not np.isfinite(
            numeric_value
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    f"Feature '{key}' contains "
                    "a non-finite value."
                ),
            )

        feature_snapshot[key] = numeric_value

    return {
        "probability": round(
            probability,
            4,
        ),

        "risk_level": risk_level,

        # NEW:
        # Exact feature vector used for the prediction.
        "features": feature_snapshot,
    }