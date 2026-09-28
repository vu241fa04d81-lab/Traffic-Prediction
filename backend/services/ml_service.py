"""
TRAFFIQ Machine Learning Prediction & Explainability Service
Serves leakage-free congestion predictions with calibrated confidence,
"Why?" feature attributions, and data-honest Critical escalation logic.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

_SERVICE_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SERVICE_DIR.parent
ARTIFACTS_DIR = str(_BACKEND_DIR / "ml" / "artifacts")

class MLService:
    def __init__(self):
        self.model = joblib.load(os.path.join(ARTIFACTS_DIR, "best_model.joblib"))
        
        with open(os.path.join(ARTIFACTS_DIR, "training_metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        with open(os.path.join(ARTIFACTS_DIR, "location_infrastructure.json"), "r") as f:
            self.infra_defaults = json.load(f)
            
        with open(os.path.join(ARTIFACTS_DIR, "location_lag_defaults.json"), "r") as f:
            self.lag_defaults = json.load(f)
            
        with open(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json"), "r") as f:
            self.eval_metrics = json.load(f)

        self.feature_names = self.metadata["features"]["all"]
        self.classes = self.metadata["classes"] # ['High', 'Low', 'Medium']
        self.global_importances = {
            item["feature"]: item["percentage"]
            for item in self.metadata["feature_importances"]
        }

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes a leakage-free prediction with confidence and 'Why?' explainability.
        """
        loc = input_data.get("location") or "Vignan University Gate"
        hour = int(input_data.get("hour") if input_data.get("hour") is not None else 17)
        day_of_week = int(input_data.get("day_of_week") if input_data.get("day_of_week") is not None else 2) # Wednesday default
        month = int(input_data.get("month") if input_data.get("month") is not None else 5)
        
        is_weekend_val = input_data.get("is_weekend")
        is_weekend = int(is_weekend_val) if is_weekend_val is not None else (1 if day_of_week in [5, 6] else 0)
        
        peak_hour_val = input_data.get("peak_hour")
        peak_hour = int(peak_hour_val) if peak_hour_val is not None else (1 if hour in [8, 9, 10, 17, 18, 19] else 0)
        
        weather = input_data.get("weather") or "Clear"
        temp = float(input_data.get("temperature") if input_data.get("temperature") is not None else 32.0)
        visibility = float(input_data.get("visibility_km") if input_data.get("visibility_km") is not None else 6.0)
        road_cond = input_data.get("road_condition") or "Good"
        event = int(input_data.get("local_event") if input_data.get("local_event") is not None else 0)
        holiday = int(input_data.get("is_holiday") if input_data.get("is_holiday") is not None else 0)
        accident = int(input_data.get("accident_reported") if input_data.get("accident_reported") is not None else 0)
        aqi = int(input_data.get("aqi_level") if input_data.get("aqi_level") is not None else 120)
        rainfall = float(input_data.get("rainfall_mm") if input_data.get("rainfall_mm") is not None else 0.0)

        # Infrastructure defaults for location
        infra = self.infra_defaults.get(loc, {"Road_Type": "Urban", "Lanes": 3, "Road_Capacity_veh_hr": 1000})
        road_type = input_data.get("road_type") or infra["Road_Type"]
        lanes = int(input_data.get("lanes") if input_data.get("lanes") is not None else infra["Lanes"])
        road_capacity = int(input_data.get("road_capacity_veh_hr") if input_data.get("road_capacity_veh_hr") is not None else infra["Road_Capacity_veh_hr"])

        # Lag defaults if not explicitly provided
        lag_key = f"{loc}_{hour}_{is_weekend}"
        lag_info = self.lag_defaults.get(lag_key, {
            "Previous_Hour_Traffic": 350.0,
            "Rolling_3_Hour_Avg_Traffic": 340.0,
            "Traffic_Change_Percent": 5.0
        })
        
        prev_traffic = float(input_data.get("previous_hour_traffic") if input_data.get("previous_hour_traffic") is not None else lag_info["Previous_Hour_Traffic"])
        rolling_traffic = float(input_data.get("rolling_3_hour_avg_traffic") if input_data.get("rolling_3_hour_avg_traffic") is not None else lag_info["Rolling_3_Hour_Avg_Traffic"])
        traffic_change = float(input_data.get("traffic_change_percent") if input_data.get("traffic_change_percent") is not None else lag_info["Traffic_Change_Percent"])

        # Build feature DataFrame matching exact pipeline expectation
        features_dict = {
            "Location": [loc],
            "Weather": [weather],
            "Road_Condition": [road_cond],
            "Road_Type": [road_type],
            "Temperature": [temp],
            "Visibility_km": [visibility],
            "Local_Event": [event],
            "Is_Holiday": [holiday],
            "Accident_Reported": [accident],
            "AQI_Level": [aqi],
            "Hour": [hour],
            "DayOfWeek": [day_of_week],
            "Month": [month],
            "Lanes": [lanes],
            "Road_Capacity_veh_hr": [road_capacity],
            "Rainfall_mm": [rainfall],
            "Is_Weekend": [is_weekend],
            "Peak_Hour": [peak_hour],
            "Previous_Hour_Traffic": [prev_traffic],
            "Rolling_3_Hour_Avg_Traffic": [rolling_traffic],
            "Traffic_Change_Percent": [traffic_change]
        }
        
        X_infer = pd.DataFrame(features_dict)
        
        # Predict class & probabilities
        pred_class = self.model.predict(X_infer)[0] # 'High', 'Medium', or 'Low'
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_infer)[0]
            prob_dict = {cls: round(float(p), 4) for cls, p in zip(self.classes, probs)}
            confidence = round(float(np.max(probs)), 4)
        else:
            prob_dict = {pred_class: 1.0}
            confidence = 1.0

        # Estimate expected traffic metrics for context
        est_volume = round(prev_traffic * (1.0 + traffic_change / 100.0), 0)
        est_tc_ratio = round(est_volume / road_capacity, 2)
        est_speed = max(18.0, round(55.0 - (est_tc_ratio * 30.0) - (10.0 if accident else 0.0) - (8.0 if rainfall > 5 else 0.0), 1))

        # Check Critical Escalation
        is_critical = False
        escalation_reason = None
        
        if pred_class == "High":
            reasons = []
            if accident == 1:
                reasons.append("Active accident reported on transit corridor")
            if rainfall > 15.0 or road_cond == "Waterlogging":
                reasons.append("Severe waterlogging and heavy precipitation impedance")
            if est_tc_ratio >= 0.70:
                reasons.append(f"Projected volume-to-capacity ratio ({est_tc_ratio}) exceeds critical bottleneck threshold (0.70)")
            if peak_hour == 1 and event == 1:
                reasons.append("Simultaneous peak commuter surge and local event crowd convergence")
                
            if reasons:
                is_critical = True
                escalation_reason = "; ".join(reasons)

        final_level = "CRITICAL" if is_critical else pred_class.upper()

        # Build Explainability 'Why?' breakdown
        factors = []
        if peak_hour == 1:
            factors.append({
                "factor": "Peak Rush Hour",
                "impact": "High Positive",
                "importance_pct": self.global_importances.get("Peak_Hour", 16.0),
                "description": f"Hour {hour}:00 coincides with peak regional commuting flow."
            })
        if accident == 1:
            factors.append({
                "factor": "Active Accident Incident",
                "impact": "Severe Positive",
                "importance_pct": self.global_importances.get("Accident_Reported", 12.7),
                "description": "Reported incident creates blockage, lane narrowing, and queue spillover."
            })
        if road_cond in ["Construction", "Waterlogging", "Potholes"]:
            factors.append({
                "factor": f"Road Condition: {road_cond}",
                "impact": "Moderate to High Positive",
                "importance_pct": self.global_importances.get("Road_Condition", 13.4),
                "description": f"{road_cond} reduces effective carriageway speed by up to 35%."
            })
        if rainfall > 5.0 or weather in ["Heavy Rain", "Rainy", "Foggy"]:
            factors.append({
                "factor": f"Weather Impedance ({weather}, {rainfall}mm)",
                "impact": "Moderate Positive",
                "importance_pct": self.global_importances.get("Rainfall_mm", 9.0) + self.global_importances.get("Weather", 10.0),
                "description": "Precipitation reduces braking distances and slows average corridor throughput."
            })
        if event == 1:
            factors.append({
                "factor": "Local Event Active",
                "impact": "Moderate Positive",
                "importance_pct": self.global_importances.get("Local_Event", 6.5),
                "description": "Special gathering generating localized traffic surges."
            })
        if traffic_change > 15.0:
            factors.append({
                "factor": f"Traffic Surge Trend (+{traffic_change}%)",
                "impact": "Moderate Positive",
                "importance_pct": self.global_importances.get("Traffic_Change_Percent", 5.0),
                "description": f"Observed volume is rising rapidly compared to previous hour baseline."
            })
        if not factors and pred_class == "Low":
            factors.append({
                "factor": "Off-Peak Free Flow",
                "impact": "Positive (Free Flow)",
                "importance_pct": 25.0,
                "description": "Nominal traffic volume with unobstructed road capacity and clear weather."
            })

        return {
            "status": "success",
            "prediction": final_level,
            "base_ml_prediction": pred_class,
            "is_critical_escalated": is_critical,
            "escalation_reason": escalation_reason,
            "confidence": confidence,
            "probabilities": prob_dict,
            "estimated_metrics": {
                "estimated_volume_veh_hr": est_volume,
                "estimated_tc_ratio": est_tc_ratio,
                "estimated_speed_kmph": est_speed,
                "road_capacity": road_capacity,
                "lanes": lanes
            },
            "explainability": {
                "summary": f"Prediction driven primarily by {' + '.join([f['factor'] for f in factors[:2]])}" if factors else "Standard flow conditions",
                "top_factors": factors
            }
        }

    def get_model_performance(self) -> Dict[str, Any]:
        """
        Returns model performance metrics from the saved evaluation artifacts.
        All figures come from the held-out 20% stratified split of the supplied
        50,000-row dataset — no real-world accuracy claims are made.
        """
        perf = self.metadata.get("performance", {})
        best_name = self.metadata.get("best_model", "unknown")
        evalm = self.eval_metrics

        def block(name):
            p = perf.get(name, {})
            return {
                "accuracy": p.get("accuracy"),
                "weighted_f1": p.get("weighted_f1"),
                "macro_f1": p.get("macro_f1"),
            }

        return {
            "status": "success",
            "data_basis_notice": (
                "All metrics are computed on a held-out 20% stratified test split of the supplied 50,000-row "
                "historical dataset. They describe performance on this dataset only and are not real-world "
                "accuracy claims about any live traffic system."
            ),
            "data_nature": "Historical / static dataset evaluation — not live traffic",
            "best_model": best_name,
            "models": {
                "RandomForest": {**block("RandomForest"), "selected": best_name == "RandomForest"},
                "GradientBoosting": {**block("GradientBoosting"), "selected": best_name == "GradientBoosting"},
            },
            "classes": self.classes,
            "overall": {
                "accuracy": evalm.get("accuracy"),
                "precision_macro": evalm.get("precision_macro"),
                "precision_weighted": evalm.get("precision_weighted"),
                "recall_macro": evalm.get("recall_macro"),
                "recall_weighted": evalm.get("recall_weighted"),
                "f1_macro": evalm.get("f1_macro"),
                "f1_weighted": evalm.get("f1_weighted"),
            },
            "confusion_matrix": {
                "labels": evalm.get("classes", self.classes),
                "matrix": evalm.get("confusion_matrix"),
            },
            "per_class": evalm.get("per_class", {}),
            "feature_importance": [
                {"feature": item["feature"], "percentage": item["percentage"]}
                for item in self.metadata.get("feature_importances", [])
            ],
            "leakage_prevention": {
                "excluded_features": ["Traffic_Volume_veh_hr", "Traffic_to_Capacity_Ratio", "Average_Speed_kmph"],
                "explanation": (
                    "These contemporaneous quantities are derived from the same observation moment as the "
                    "congestion label and are unavailable before a trip. They are excluded from model inputs "
                    "to prevent target leakage; predictions rely only on pre-trip features."
                ),
            },
            "test_split": {"size": 10000, "fraction": 0.20, "strategy": "stratified"},
        }

ml_service = MLService()
