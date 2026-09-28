"""
TRAFFIQ Backend API
FastAPI Application for Smart City Traffic Intelligence & Route Analytics.
"""

import sys
from pathlib import Path

# Ensure both backend directory and project root are in sys.path
_BACKEND_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _BACKEND_DIR.parent
for _p in [str(_BACKEND_DIR), str(_ROOT_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

try:
    from backend.services.geospatial_service import geospatial_service
    from backend.services.analytics_service import analytics_service
    from backend.services.ml_service import ml_service
    from backend.services.recommendation_service import recommendation_service
    from backend.services.simulation_service import simulation_service
    from backend.services.route_intelligence_service import route_intelligence_service
    from backend.services.dataset_service import dataset_service  # custom CSV upload analysis + geocoding only
except ImportError:
    from services.geospatial_service import geospatial_service
    from services.analytics_service import analytics_service
    from services.ml_service import ml_service
    from services.recommendation_service import recommendation_service
    from services.simulation_service import simulation_service
    from services.route_intelligence_service import route_intelligence_service
    from services.dataset_service import dataset_service  # custom CSV upload analysis + geocoding only

app = FastAPI(
    title="TRAFFIQ API",
    description="Smart City Traffic Congestion Intelligence & Route Analytics",
    version="1.0.0"
)

# CORS configuration
# Note: allow_origins=["*"] is incompatible with allow_credentials=True per the CORS
# spec (browsers reject the response when both are set). Starlette mitigates at runtime,
# but we keep the config valid: wildcard origins with credentials disabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins during hackathon development
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- PYDANTIC SCHEMAS -----------------

class PredictionRequest(BaseModel):
    location: str = Field(default="Vignan University Gate")
    hour: int = Field(default=17, ge=0, le=23)
    day_of_week: int = Field(default=2, ge=0, le=6)
    month: int = Field(default=5, ge=1, le=12)
    weather: str = Field(default="Clear")
    temperature: float = Field(default=32.0)
    visibility_km: float = Field(default=6.0)
    road_condition: str = Field(default="Good")
    local_event: int = Field(default=0, ge=0, le=1)
    is_holiday: int = Field(default=0, ge=0, le=1)
    accident_reported: int = Field(default=0, ge=0, le=1)
    aqi_level: int = Field(default=120)
    rainfall_mm: float = Field(default=0.0)
    is_weekend: Optional[int] = None
    peak_hour: Optional[int] = None
    previous_hour_traffic: Optional[float] = None
    rolling_3_hour_avg_traffic: Optional[float] = None
    traffic_change_percent: Optional[float] = None

class RouteRequest(BaseModel):
    origin: str = Field(default="Vignan University Gate")
    destination: str = Field(default="City Center")

class RoutePointsRequest(BaseModel):
    origin_lat: float = Field(..., ge=-90, le=90)
    origin_lon: float = Field(..., ge=-180, le=180)
    dest_lat: float = Field(..., ge=-90, le=90)
    dest_lon: float = Field(..., ge=-180, le=180)
    origin_name: Optional[str] = Field(default=None)
    destination_name: Optional[str] = Field(default=None)

class RouteIntelligenceRequest(BaseModel):
    origin: str = Field(default="Vignan University Gate")
    destination: str = Field(default="City Center")
    hour: int = Field(default=17, ge=0, le=23)
    day_of_week: int = Field(default=2, ge=0, le=6)
    month: int = Field(default=5, ge=1, le=12)
    weather: str = Field(default="Clear")
    temperature: float = Field(default=32.0)
    visibility_km: float = Field(default=6.0)
    road_condition: str = Field(default="Good")
    local_event: int = Field(default=0, ge=0, le=1)
    is_holiday: int = Field(default=0, ge=0, le=1)
    accident_reported: int = Field(default=0, ge=0, le=1)
    rainfall_mm: float = Field(default=0.0)

class RecommendationRequest(BaseModel):
    prediction_result: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

class SimulationRequest(BaseModel):
    location: str = Field(default="Vignan University Gate")
    volume_delta_percent: float = Field(default=20.0)
    accident_injected: int = Field(default=0, ge=0, le=1)
    weather_override: str = Field(default="Clear")
    rainfall_override: float = Field(default=0.0)
    road_condition_override: str = Field(default="Good")
    hour: int = Field(default=18, ge=0, le=23)
    day_of_week: int = Field(default=4, ge=0, le=6)
    month: int = Field(default=5, ge=1, le=12)
    local_event: int = Field(default=0, ge=0, le=1)
    is_holiday: int = Field(default=0, ge=0, le=1)

# ----------------- API ENDPOINTS -----------------

@app.get("/")
def read_root():
    return {
        "project": "TRAFFIQ",
        "tagline": "Smart Traffic Congestion Intelligence & Route Analytics",
        "status": "online",
        "version": "1.0.0",
        "monitored_region": "Andhra Pradesh Urban Corridor (Vignan / Guntur / Vijayawada)",
        "endpoints": [
            "/api/geocode",
            "/api/route/points",
            "/api/locations",
            "/api/analytics/summary",
            "/api/analytics/location/{location}",
            "/api/dataset/overview",
            "/api/dataset/preview",
            "/api/dataset/analytics/full",
            "/api/dataset/upload",
            "/api/model/performance",
            "/api/predict",
            "/api/route",
            "/api/route-intelligence",
            "/api/recommendation",
            "/api/simulation"
        ]
    }

@app.get("/api/locations")
def get_locations():
    """Returns verified dataset locations with geocoded coordinates and infrastructure attributes."""
    return geospatial_service.get_all_locations()

@app.get("/api/analytics/summary")
def get_analytics_summary():
    """Returns global Level 1 traffic analytics, distributions, and trends."""
    return analytics_service.get_summary()

@app.get("/api/analytics/location/{location}")
def get_location_analytics(location: str):
    """Returns granular analytics for a specific monitored corridor location."""
    res = analytics_service.get_location_analytics(location)
    if res.get("status") == "error":
        raise HTTPException(status_code=404, detail=res["message"])
    return res

# ----------------- DATASET & MODEL ENDPOINTS -----------------

@app.get("/api/dataset/overview")
def get_dataset_overview():
    """Automatic schema inspection of the default dataset: rows, columns, types, missing values, duplicates, ranges."""
    try:
        return analytics_service.get_dataset_overview()
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/preview")
def get_dataset_preview(search: str = "", limit: int = 20):
    """Dataset preview rows with optional case-insensitive text search across text columns."""
    try:
        return analytics_service.get_dataset_preview(search or None, limit)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/dataset/analytics/full")
def get_full_analytics():
    """All analytics sections generated automatically from the default dataset (historical/static data)."""
    try:
        return {
            "status": "success",
            "data_nature": "Historical / static dataset — not live or real-time traffic",
            "analytics": analytics_service.get_dataset_analytics(),
        }
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/dataset/upload")
def upload_dataset(file: UploadFile = File(...)):
    """Analyze an uploaded compatible traffic CSV: schema, automatic analytics, ML feature availability, geography. The default dataset remains the active demo dataset."""
    try:
        contents = file.file.read()
        return dataset_service.analyze_upload(contents, file.filename or "upload.csv")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload analysis failed: {e}")

@app.get("/api/model/performance")
def get_model_performance():
    """Model performance from saved evaluation artifacts (held-out split of the supplied dataset only)."""
    return ml_service.get_model_performance()

@app.post("/api/predict")
def predict_congestion(req: PredictionRequest):
    """Predicts congestion level (LOW/MEDIUM/HIGH/CRITICAL) with explainability."""
    payload = req.model_dump()
    res = ml_service.predict(payload)
    # Also attach contextual recommendations
    recs = recommendation_service.generate_recommendations(res, payload)
    res["recommendations"] = recs
    return res

@app.post("/api/route")
def get_route(req: RouteRequest):
    """Returns legitimate OSRM road route geometry and routing metrics."""
    res = geospatial_service.get_route(req.origin, req.destination)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res["message"])
    return res

# ----------------- GLOBAL GEOCODE SEARCH (any real-world place) -----------------
# Reuses the single existing Nominatim geocoder (geospatial_service.geocode_location)
# and the dataset_service persistent cache. No new geocoding system is introduced.

@app.get("/api/geocode")
def geocode_place(q: str):
    """Geocode any real-world place via OpenStreetMap Nominatim (cached).
    Returns real coordinates only; never traffic data. Dataset membership is returned as a flag.
    """
    query = (q or "").strip()
    if not query:
        raise HTTPException(status_code=400, detail="Missing search query")
    res = dataset_service.geocode_single(query)
    if res["status"] == "not_found":
        return {"status": "not_found", "query": query, "lat": None, "lon": None, "display_name": None, "in_dataset": False}
    if res["status"] != "ok":
        raise HTTPException(status_code=502, detail=res.get("message") or "Geocoding service unavailable")
    is_dataset = geospatial_service.get_location(query) is not None
    return {
        "status": "ok",
        "query": query,
        "lat": res["lat"],
        "lon": res["lon"],
        "display_name": res["display_name"],
        "in_dataset": is_dataset,
        "dataset_location": geospatial_service.get_location(query) if is_dataset else None,
    }


@app.post("/api/route/points")
def get_route_points(req: RoutePointsRequest):
    """Road route between arbitrary geocoded coordinates via the existing OSRM driving-profile integration.
    Same OSRM call pattern and response shape as /api/route — routing and estimated duration only,
    never live congestion. Coordinates must come from the geocoding endpoint, not user guesswork.
    """
    res = geospatial_service.get_route_by_coords(
        req.origin_lat, req.origin_lon,
        req.dest_lat, req.dest_lon,
        origin_name=req.origin_name,
        dest_name=req.destination_name,
    )
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res["message"])
    return res

@app.post("/api/route-intelligence")
def get_route_intelligence(req: RouteIntelligenceRequest):
    """Synthesizes road route geometry with origin/destination historical context and ML predictions."""
    res = route_intelligence_service.analyze_route(req.model_dump())
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res["message"])
    return res

@app.post("/api/recommendation")
def get_recommendations(req: RecommendationRequest):
    """Generates separated User Travel Advice and Traffic Management Recommendations."""
    return recommendation_service.generate_recommendations(
        req.prediction_result,
        req.context or {}
    )

@app.post("/api/simulation")
def run_simulation(req: SimulationRequest):
    """Runs a What-If scenario sensitivity stress test with consistent metric updates."""
    return simulation_service.simulate_scenario(req.model_dump())

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8001, reload=True)
