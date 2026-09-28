import sys
from pathlib import Path

_TEST_DIR = Path(__file__).resolve().parent
for _p in [str(_TEST_DIR), str(_TEST_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from starlette.testclient import TestClient

try:
    from backend.main import app
except ImportError:
    from main import app

client = TestClient(app)

def test_endpoints():
    print("Testing GET / ...")
    r = client.get("/")
    assert r.status_code == 200, f"Root failed: {r.text}"
    print(" Root OK:", r.json()["project"])

    print("Testing GET /api/locations ...")
    r = client.get("/api/locations")
    assert r.status_code == 200
    locations = r.json()
    assert len(locations) == 7
    print(f" Locations OK: {len(locations)} verified locations found.")

    print("Testing GET /api/analytics/summary ...")
    r = client.get("/api/analytics/summary")
    assert r.status_code == 200
    summary = r.json()
    print(f" Summary OK: {summary['total_records']} records, avg volume: {summary['kpis']['avg_traffic_volume']}")

    print("Testing GET /api/analytics/location/Vignan University Gate ...")
    r = client.get("/api/analytics/location/Vignan University Gate")
    assert r.status_code == 200
    loc_data = r.json()
    print(f" Location Analytics OK: {loc_data['location_name']}, capacity: {loc_data['capacity']}")

    print("Testing POST /api/predict ...")
    pred_req = {
        "location": "Vignan University Gate",
        "hour": 18,
        "day_of_week": 4, # Friday
        "weather": "Heavy Rain",
        "rainfall_mm": 18.5,
        "road_condition": "Waterlogging",
        "accident_reported": 1,
        "peak_hour": 1
    }
    r = client.post("/api/predict", json=pred_req)
    assert r.status_code == 200
    pred = r.json()
    print(f" Predict OK: {pred['prediction']} (Base: {pred['base_ml_prediction']}, Critical: {pred['is_critical_escalated']})")
    print(f" Confidence: {pred['confidence']}, Top factor: {pred['explainability']['top_factors'][0]['factor']}")

    print("Testing POST /api/route ...")
    route_req = {
        "origin": "Vignan University Gate",
        "destination": "City Center"
    }
    r = client.post("/api/route", json=route_req)
    assert r.status_code == 200
    route = r.json()
    print(f" Route OK: {route['distance_km']} km, {route['duration_min']} min, {len(route['geometry']['coordinates'])} points.")

    print("Testing POST /api/route-intelligence ...")
    route_intel_req = {
        "origin": "Vignan University Gate",
        "destination": "City Center",
        "hour": 18,
        "weather": "Clear"
    }
    r = client.post("/api/route-intelligence", json=route_intel_req)
    assert r.status_code == 200
    intel = r.json()
    print(f" Route Intelligence OK: Departure pred: {intel['origin_intelligence']['departure_ml_prediction']}, Arrival pred: {intel['destination_intelligence']['arrival_ml_prediction']}")

    print("Testing POST /api/simulation ...")
    sim_req = {
        "location": "Vignan University Gate",
        "volume_delta_percent": 30.0,
        "accident_injected": 1
    }
    r = client.post("/api/simulation", json=sim_req)
    assert r.status_code == 200
    sim = r.json()
    print(f" Simulation OK: Base: {sim['comparison']['baseline']['congestion_level']} -> Sim: {sim['comparison']['simulated']['congestion_level']}")

    print("\nALL BACKEND API TESTS PASSED PERFECTLY!")

if __name__ == "__main__":
    test_endpoints()
