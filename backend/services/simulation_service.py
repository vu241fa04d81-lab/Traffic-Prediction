"""
TRAFFIQ Scenario Simulation Service
Executes 'What-If' sensitivity simulations by modulating baseline demand (+10%, +25%, +50%, incidents, weather)
and recalculating derived capacity metrics consistently.
Clearly labeled as 'Scenario Simulation' - never live traffic.
"""

from typing import Dict, Any, List

try:
    from backend.services.ml_service import ml_service
    from backend.services.recommendation_service import recommendation_service
except ImportError:
    from services.ml_service import ml_service
    from services.recommendation_service import recommendation_service

class SimulationService:
    def simulate_scenario(self, scenario_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Runs what-if simulation:
        1. Modulates volume delta percent (e.g. +10%, +25%, -20%)
        2. Modulates incident / weather injection
        3. Recalculates derived features consistently
        4. Evaluates ML prediction
        5. Generates simulation-specific travel & traffic control recommendations
        """
        loc = scenario_input.get("location", "Vignan University Gate")
        volume_delta_pct = float(scenario_input.get("volume_delta_percent", 20.0))
        accident_injected = int(scenario_input.get("accident_injected", 0))
        weather_override = scenario_input.get("weather_override", "Clear")
        rainfall_override = float(scenario_input.get("rainfall_override", 0.0))
        road_cond_override = scenario_input.get("road_condition_override", "Good")
        hour = int(scenario_input.get("hour", 18))
        day_of_week = int(scenario_input.get("day_of_week", 4)) # Friday
        is_weekend = 1 if day_of_week in [5, 6] else 0
        peak_hour = 1 if hour in [8, 9, 10, 17, 18, 19] else 0

        # Baseline lookup
        lag_key = f"{loc}_{hour}_{is_weekend}"
        base_lag = ml_service.lag_defaults.get(lag_key, {
            "Previous_Hour_Traffic": 350.0,
            "Rolling_3_Hour_Avg_Traffic": 340.0,
            "Traffic_Change_Percent": 5.0
        })
        infra = ml_service.infra_defaults.get(loc, {"Road_Type": "Urban", "Lanes": 3, "Road_Capacity_veh_hr": 1000})

        base_prev = base_lag["Previous_Hour_Traffic"]
        sim_prev = round(base_prev * (1.0 + volume_delta_pct / 100.0), 1)
        sim_rolling = round(base_lag["Rolling_3_Hour_Avg_Traffic"] * (1.0 + (volume_delta_pct * 0.7) / 100.0), 1)
        sim_traffic_change = round(base_lag["Traffic_Change_Percent"] + volume_delta_pct, 1)

        sim_payload = {
            "location": loc,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": int(scenario_input.get("month", 5)),
            "is_weekend": is_weekend,
            "peak_hour": peak_hour,
            "weather": weather_override,
            "temperature": float(scenario_input.get("temperature", 31.0)),
            "visibility_km": float(scenario_input.get("visibility_km", 6.0 if weather_override == "Clear" else 2.5)),
            "road_condition": road_cond_override,
            "local_event": int(scenario_input.get("local_event", 0)),
            "is_holiday": int(scenario_input.get("is_holiday", 0)),
            "accident_reported": accident_injected,
            "aqi_level": int(scenario_input.get("aqi_level", 140)),
            "rainfall_mm": rainfall_override,
            "previous_hour_traffic": sim_prev,
            "rolling_3_hour_avg_traffic": sim_rolling,
            "traffic_change_percent": sim_traffic_change,
            "road_type": infra["Road_Type"],
            "lanes": infra["Lanes"],
            "road_capacity_veh_hr": infra["Road_Capacity_veh_hr"]
        }

        # Predict with ML
        pred_res = ml_service.predict(sim_payload)

        # Baseline comparison (0% delta, no incident, clear weather)
        base_payload = sim_payload.copy()
        base_payload["previous_hour_traffic"] = base_prev
        base_payload["rolling_3_hour_avg_traffic"] = base_lag["Rolling_3_Hour_Avg_Traffic"]
        base_payload["traffic_change_percent"] = base_lag["Traffic_Change_Percent"]
        base_payload["accident_reported"] = 0
        base_payload["weather"] = "Clear"
        base_payload["rainfall_mm"] = 0.0
        base_payload["road_condition"] = "Good"
        base_res = ml_service.predict(base_payload)

        # Actionable recommendations
        recs = recommendation_service.generate_recommendations(pred_res, sim_payload)

        return {
            "status": "success",
            "simulation_mode": "Scenario Simulation (Synthetic Stress Test)",
            "disclaimer": "This is a mathematical scenario simulation for urban stress testing. NOT live traffic observations.",
            "inputs": {
                "location": loc,
                "volume_delta_percent": volume_delta_pct,
                "accident_injected": bool(accident_injected),
                "weather": weather_override,
                "rainfall_mm": rainfall_override,
                "road_condition": road_cond_override,
                "hour": hour,
                "day_of_week": day_of_week
            },
            "comparison": {
                "baseline": {
                    "congestion_level": base_res["prediction"],
                    "estimated_volume": base_res["estimated_metrics"]["estimated_volume_veh_hr"],
                    "tc_ratio": base_res["estimated_metrics"]["estimated_tc_ratio"],
                    "estimated_speed_kmph": base_res["estimated_metrics"]["estimated_speed_kmph"]
                },
                "simulated": {
                    "congestion_level": pred_res["prediction"],
                    "estimated_volume": pred_res["estimated_metrics"]["estimated_volume_veh_hr"],
                    "tc_ratio": pred_res["estimated_metrics"]["estimated_tc_ratio"],
                    "estimated_speed_kmph": pred_res["estimated_metrics"]["estimated_speed_kmph"],
                    "is_critical_escalated": pred_res["is_critical_escalated"],
                    "escalation_reason": pred_res.get("escalation_reason")
                },
                "deltas": {
                    "volume_change_veh_hr": round(pred_res["estimated_metrics"]["estimated_volume_veh_hr"] - base_res["estimated_metrics"]["estimated_volume_veh_hr"], 1),
                    "speed_loss_kmph": round(base_res["estimated_metrics"]["estimated_speed_kmph"] - pred_res["estimated_metrics"]["estimated_speed_kmph"], 1),
                    "tc_ratio_increase": round(pred_res["estimated_metrics"]["estimated_tc_ratio"] - base_res["estimated_metrics"]["estimated_tc_ratio"], 2)
                }
            },
            "explainability": pred_res["explainability"],
            "recommendations": recs
        }

simulation_service = SimulationService()
