"""
TRAFFIQ Preprocessing Pipeline
Defines leakage-free feature sets and scikit-learn transformers.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Target variable
TARGET_COL = "Congestion_Level"

# Leakage-free features available PRE-TRIP / FORECAST period
CATEGORICAL_FEATURES = [
    "Location",
    "Weather",
    "Road_Condition",
    "Road_Type"
]

NUMERICAL_FEATURES = [
    "Temperature",
    "Visibility_km",
    "Local_Event",
    "Is_Holiday",
    "Accident_Reported",
    "AQI_Level",
    "Hour",
    "DayOfWeek",
    "Month",
    "Lanes",
    "Road_Capacity_veh_hr",
    "Rainfall_mm",
    "Is_Weekend",
    "Peak_Hour",
    "Previous_Hour_Traffic",
    "Rolling_3_Hour_Avg_Traffic",
    "Traffic_Change_Percent"
]

ALL_MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

# Contemporaneous features excluded from pre-trip prediction to prevent target leakage:
CONTEMPORANEOUS_LEAKAGE_FEATURES = [
    "Traffic_Volume_veh_hr",
    "Traffic_to_Capacity_Ratio",
    "Average_Speed_kmph"
]

def build_preprocessor() -> ColumnTransformer:
    """Builds a scikit-learn ColumnTransformer for categorical and numerical features."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES
            ),
            (
                "num",
                StandardScaler(),
                NUMERICAL_FEATURES
            )
        ],
        remainder="drop"
    )
    return preprocessor

def get_location_lag_defaults(df: pd.DataFrame) -> dict:
    """
    Computes baseline historical median lag features grouped by (Location, Hour, Is_Weekend)
    so users can predict without having to manually invent previous hour traffic numbers.
    """
    grouped = df.groupby(["Location", "Hour", "Is_Weekend"])[
        ["Previous_Hour_Traffic", "Rolling_3_Hour_Avg_Traffic", "Traffic_Change_Percent"]
    ].median().reset_index()
    
    defaults = {}
    for _, row in grouped.iterrows():
        key = f"{row['Location']}_{int(row['Hour'])}_{int(row['Is_Weekend'])}"
        defaults[key] = {
            "Previous_Hour_Traffic": float(row["Previous_Hour_Traffic"]),
            "Rolling_3_Hour_Avg_Traffic": float(row["Rolling_3_Hour_Avg_Traffic"]),
            "Traffic_Change_Percent": float(row["Traffic_Change_Percent"])
        }
    return defaults

def get_location_infrastructure_defaults(df: pd.DataFrame) -> dict:
    """
    Extracts fixed road infrastructure attributes per location:
    Road_Type, Lanes, Road_Capacity_veh_hr
    """
    grouped = df.groupby("Location")[["Road_Type", "Lanes", "Road_Capacity_veh_hr"]].first()
    return grouped.to_dict(orient="index")
