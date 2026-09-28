"""TRAFFIQ Backend API - Extended Endpoint Tests
Covers the automatic dataset analytics, model performance, and upload analysis endpoints.
Run: py test_endpoints_extended.py
"""

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


def test_dataset_overview():
    print("Testing GET /api/dataset/overview ...")
    r = client.get("/api/dataset/overview")
    assert r.status_code == 200, r.text
    ov = r.json()
    assert ov["row_count"] == 50000
    assert ov["column_count"] == 30
    assert ov["unique_locations"] == 7
    assert ov["missing_cells"] == 0
    assert ov["duplicate_rows"] == 0
    assert ov["timestamp_range"]["available"] is True
    assert len(ov["columns"]) == 30
    assert any(c["name"] == "Congestion_Level" for c in ov["columns"])
    print(f" Overview OK: {ov['row_count']} rows, {ov['column_count']} cols, {ov['unique_locations']} locations, range {ov['timestamp_range']['start']} -> {ov['timestamp_range']['end']}")


def test_full_analytics():
    print("Testing GET /api/dataset/analytics/full ...")
    r = client.get("/api/dataset/analytics/full")
    assert r.status_code == 200, r.text
    a = r.json()["analytics"]

    # All 15 sections available for the default dataset
    expected = [
        "congestion_distribution", "volume_by_hour", "congestion_by_hour",
        "location_congestion", "weather_congestion", "road_condition_congestion",
        "weekday_weekend", "accident_congestion", "event_congestion",
        "hour_location_heatmap", "volume_trend", "speed_trend",
        "tcr_analysis", "corridor_comparison", "numeric_correlations",
    ]
    for key in expected:
        assert key in a, f"missing section {key}"
        assert a[key].get("available") is True, f"{key} not available: {a[key].get('reason')}"

    cd = a["congestion_distribution"]
    assert abs(cd["percentages"]["Low"] - 52.74) < 0.01
    assert abs(cd["percentages"]["Medium"] - 36.72) < 0.01
    assert abs(cd["percentages"]["High"] - 10.54) < 0.01
    assert len(a["hour_location_heatmap"]["hours"]) == 24
    assert len(a["hour_location_heatmap"]["locations"]) == 7
    assert len(a["volume_by_hour"]["hours"]) == 24
    assert len(a["volume_trend"]["dates"]) > 300
    assert a["weekday_weekend"]["rows"][0]["segment"] == "Weekday"
    assert a["corridor_comparison"]["highest_risk"] is not None
    assert len(a["numeric_correlations"]["rows"]) > 0
    print(f" Full analytics OK: all {len(expected)} sections available; highest-risk corridor: {a['corridor_comparison']['highest_risk']['location']}")


def test_dataset_preview_search():
    print("Testing GET /api/dataset/preview ...")
    r = client.get("/api/dataset/preview?limit=5")
    assert r.status_code == 200
    p = r.json()
    assert len(p["rows"]) == 5 and len(p["columns"]) == 30
    print(f" Preview OK: {p['returned']} rows of {p['dataset_total']}")

    r = client.get("/api/dataset/preview?search=MG%20Road&limit=5")
    assert r.status_code == 200
    p = r.json()
    assert p["matched_total"] > 0
    print(f" Search OK: 'MG Road' matched {p['matched_total']} rows")


def test_model_performance():
    print("Testing GET /api/model/performance ...")
    r = client.get("/api/model/performance")
    assert r.status_code == 200
    mp = r.json()
    assert mp["best_model"] == "GradientBoosting"
    assert abs(mp["overall"]["accuracy"] - 0.9996) < 0.0001
    assert mp["confusion_matrix"]["labels"] == ["High", "Low", "Medium"]
    assert len(mp["confusion_matrix"]["matrix"]) == 3
    assert len(mp["feature_importance"]) == 21
    assert "not real-world" in mp["data_basis_notice"]
    assert len(mp["leakage_prevention"]["excluded_features"]) == 3
    print(f" Model performance OK: best={mp['best_model']}, accuracy={mp['overall']['accuracy']}, leakage-excluded={mp['leakage_prevention']['excluded_features']}")


def test_upload_analysis():
    print("Testing POST /api/dataset/upload (default schema subset) ...")
    import io
    import pandas as pd
    df = pd.read_csv(str(_TEST_DIR / "data" / "traffic_processed.csv")).head(2000)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    r = client.post("/api/dataset/upload", files={"file": ("sample.csv", buf, "text/csv")})
    assert r.status_code == 200, r.text
    res = r.json()
    assert res["status"] == "success"
    assert res["overview"]["row_count"] == 2000
    # Processed schema carries location NAMES (no lat/lon), so Nominatim geocoding is the expected path
    assert res["geography"]["method"] == "OpenStreetMap Nominatim geocoding of location names"
    assert res["ml_feature_availability"]["target_available"] is True
    assert res["ml_feature_availability"]["missing_required"] == []
    a = res["analytics"]
    assert a["congestion_distribution"]["available"] is True
    assert a["hour_location_heatmap"]["available"] is True
    print(f" Upload (full schema) OK: {res['overview']['row_count']} rows, geography={res['geography']['method']} ({res['geography']['geocoded'] if 'geocoded' in res['geography'] else len(res['geography']['locations'])} geocoded)")

    print("Testing POST /api/dataset/upload (lat/lon provided directly) ...")
    df_geo = df.copy()
    df_geo["Latitude"] = 16.3
    df_geo["Longitude"] = 80.44
    buf3 = io.BytesIO()
    df_geo.to_csv(buf3, index=False)
    buf3.seek(0)
    r = client.post("/api/dataset/upload", files={"file": ("withgeo.csv", buf3, "text/csv")})
    assert r.status_code == 200, r.text
    res3 = r.json()
    assert res3["geography"]["available"] is True
    assert res3["geography"]["method"] == "latitude/longitude columns used directly (no fabrication)"
    assert len(res3["geography"]["locations"]) > 0
    print(f" Upload (lat/lon) OK: {len(res3['geography']['locations'])} unique coordinate pairs used directly")

    print("Testing POST /api/dataset/upload (renamed columns, no geography) ...")
    df2 = df.rename(columns={
        "Traffic_Volume_veh_hr": "Vehicle_Count",
        "Average_Speed_kmph": "Speed_kmph",
        "Congestion_Level": "Traffic_Condition",
    }).drop(columns=["Location_ID", "Latitude", "Longitude"], errors="ignore")
    buf2 = io.BytesIO()
    df2.to_csv(buf2, index=False)
    buf2.seek(0)
    r = client.post("/api/dataset/upload", files={"file": ("renamed.csv", buf2, "text/csv")})
    assert r.status_code == 200, r.text
    res2 = r.json()
    assert res2["analytics"]["congestion_distribution"]["available"] is True  # via Traffic_Condition alias
    assert res2["analytics"]["volume_by_hour"]["available"] is True           # via Vehicle_Count alias
    # No lat/lon -> tries geocoding of location names (may succeed or fail depending on network)
    assert res2["geography"]["available"] in (True, False)
    print(f" Upload (aliased schema) OK: congestion via alias, geography notice: {res2['geography'].get('notice') or res2['geography']['method']}")


if __name__ == "__main__":
    test_dataset_overview()
    test_full_analytics()
    test_dataset_preview_search()
    test_model_performance()
    test_upload_analysis()
    print("\nALL EXTENDED ENDPOINT TESTS PASSED!")
