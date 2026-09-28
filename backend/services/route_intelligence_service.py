"""
TRAFFIQ Route Intelligence Service
Combines real OSRM road geometry with dataset historical analytics and ML predictions.
Upholds strict data honesty: distinguishes endpoint location analytics from corridor routing.
"""

from typing import Dict, Any

try:
    from backend.services.geospatial_service import geospatial_service
    from backend.services.analytics_service import analytics_service
    from backend.services.ml_service import ml_service
    from backend.services.recommendation_service import recommendation_service
except ImportError:
    from services.geospatial_service import geospatial_service
    from services.analytics_service import analytics_service
    from services.ml_service import ml_service
    from services.recommendation_service import recommendation_service

class RouteIntelligenceService:
    def analyze_route(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        origin_key = request_payload.get("origin", "Vignan University Gate")
        dest_key = request_payload.get("destination", "City Center")
        
        # 1. Fetch real OSRM road geometry
        route_res = geospatial_service.get_route(origin_key, dest_key)
        if route_res.get("status") != "success":
            return route_res

        # 2. Extract historical context for origin & destination
        origin_hist = analytics_service.get_location_analytics(origin_key)
        dest_hist = analytics_service.get_location_analytics(dest_key)

        # 3. Predict departure corridor congestion (ML)
        ml_input = request_payload.copy()
        ml_input["location"] = route_res["origin"]["name"]
        pred_res = ml_service.predict(ml_input)

        # 4. Predict arrival corridor congestion (ML)
        dest_ml_input = request_payload.copy()
        dest_ml_input["location"] = route_res["destination"]["name"]
        # adjust hour by estimated travel duration
        travel_hours = max(0, int(round(route_res["duration_min"] / 60.0)))
        dest_ml_input["hour"] = min(23, int(request_payload.get("hour", 17)) + travel_hours)
        dest_pred_res = ml_service.predict(dest_ml_input)

        # 5. Generate Recommendations
        recs = recommendation_service.generate_recommendations(pred_res, request_payload)

        # 6. Synthesize comprehensive Route Intelligence
        return {
            "status": "success",
            "intelligence_type": "Dataset-Informed Route Intelligence",
            "data_honesty_notice": "Road geometry is provided by OpenStreetMap / OSRM. Congestion levels reflect dataset-monitored corridor endpoints and ML predictive modeling, not real-time per-meter sensor feeds.",
            "route": {
                "origin": route_res["origin"],
                "destination": route_res["destination"],
                "distance_km": route_res["distance_km"],
                "estimated_duration_min": route_res["duration_min"],
                "geometry": route_res["geometry"],
                "source": route_res["source"]
            },
            "origin_intelligence": {
                "location_name": route_res["origin"]["name"],
                "historical_kpis": origin_hist.get("kpis", {}),
                "historical_congestion_distribution": origin_hist.get("congestion_distribution", {}).get("percentages", {}),
                "departure_ml_prediction": pred_res["prediction"],
                "base_ml_prediction": pred_res["base_ml_prediction"],
                "is_critical_escalated": pred_res["is_critical_escalated"],
                "confidence": pred_res["confidence"],
                "estimated_traffic_volume": pred_res["estimated_metrics"]["estimated_volume_veh_hr"],
                "estimated_speed_kmph": pred_res["estimated_metrics"]["estimated_speed_kmph"],
                "tc_ratio": pred_res["estimated_metrics"]["estimated_tc_ratio"]
            },
            "destination_intelligence": {
                "location_name": route_res["destination"]["name"],
                "historical_kpis": dest_hist.get("kpis", {}),
                "historical_congestion_distribution": dest_hist.get("congestion_distribution", {}).get("percentages", {}),
                "arrival_ml_prediction": dest_pred_res["prediction"],
                "estimated_traffic_volume": dest_pred_res["estimated_metrics"]["estimated_volume_veh_hr"],
                "estimated_speed_kmph": dest_pred_res["estimated_metrics"]["estimated_speed_kmph"],
                "tc_ratio": dest_pred_res["estimated_metrics"]["estimated_tc_ratio"]
            },
            "explainability": pred_res["explainability"],
            "recommendations": recs
        }

route_intelligence_service = RouteIntelligenceService()
