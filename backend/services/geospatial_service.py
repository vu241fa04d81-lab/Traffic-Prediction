"""
TRAFFIQ Geospatial Service
Manages verified locations, Nominatim metadata, and OSRM road routing with local caching.
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SERVICE_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SERVICE_DIR.parent
LOCATIONS_PATH = str(_BACKEND_DIR / "data" / "locations.csv")
CACHE_PATH = str(_BACKEND_DIR / "data" / "cache" / "routes_cache.json")
OSRM_BASE_URL = "https://router.project-osrm.org/route/v1/driving"
NOMINATIM_BASE_URL = "https://nominatim.openstreetmap.org/search"


def geocode_location(name: str) -> dict:
    """
    Geocode a location name via OpenStreetMap Nominatim.
    Returns a cached-free, single-shot result: {status, lat, lon, display_name}.
    (Persistent caching and rate limiting live in dataset_service.geocode_locations.)
    """
    headers = {
        "User-Agent": "TRAFFIQ-Academic-SmartCity/1.0 (academic; smart-city-analytics)"
    }
    try:
        resp = requests.get(
            NOMINATIM_BASE_URL,
            params={"q": name, "format": "jsonv2", "limit": 1, "addressdetails": 0},
            headers=headers,
            timeout=10,
        )
        if resp.status_code == 200:
            results = resp.json()
            if results:
                first = results[0]
                return {
                    "status": "ok",
                    "lat": float(first["lat"]),
                    "lon": float(first["lon"]),
                    "display_name": first.get("display_name"),
                }
            return {"status": "not_found", "lat": None, "lon": None, "display_name": None}
        return {"status": "error", "lat": None, "lon": None, "display_name": None, "message": f"Nominatim HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "lat": None, "lon": None, "display_name": None, "message": str(e)}

class GeospatialService:
    def __init__(self):
        self.locations_df = pd.read_csv(LOCATIONS_PATH)
        self.locations_dict = {}
        for _, row in self.locations_df.iterrows():
            item = row.to_dict()
            self.locations_dict[item["Location_Name"]] = item
            self.locations_dict[item["Location_ID"]] = item
            
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if os.path.exists(CACHE_PATH):
            try:
                with open(CACHE_PATH, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        try:
            with open(CACHE_PATH, "w") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to persist route cache: {e}")

    def get_all_locations(self) -> List[dict]:
        """Returns all verified dataset locations."""
        return self.locations_df.to_dict(orient="records")

    def get_location(self, key: str) -> Optional[dict]:
        """Lookup by Location_Name or Location_ID."""
        return self.locations_dict.get(key)

    def get_route(self, origin_key: str, dest_key: str) -> dict:
        """
        Calculates legitimate road route between two dataset locations using OSRM.
        Checks local cache first.
        """
        origin = self.get_location(origin_key)
        dest = self.get_location(dest_key)

        if not origin:
            return {"status": "error", "message": f"Origin location '{origin_key}' not found in verified master"}
        if not dest:
            return {"status": "error", "message": f"Destination location '{dest_key}' not found in verified master"}
        if origin["Location_ID"] == dest["Location_ID"]:
            return {
                "status": "error",
                "message": "Origin and destination cannot be the same location"
            }

        cache_key = f"{origin['Location_ID']}->{dest['Location_ID']}"
        if cache_key in self.cache:
            res = self.cache[cache_key].copy()
            res["source"] = "local_cache"
            return res

        # Call live OSRM HTTPS API
        # OSRM coordinate syntax: {lon1},{lat1};{lon2},{lat2}
        lon1, lat1 = origin["Longitude"], origin["Latitude"]
        lon2, lat2 = dest["Longitude"], dest["Latitude"]
        url = f"{OSRM_BASE_URL}/{lon1},{lat1};{lon2},{lat2}?overview=full&geometries=geojson"
        
        headers = {
            "User-Agent": "TRAFFIQ-Academic-SmartCity/1.0 (academic; smart-city-analytics)"
        }

        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    primary_route = data["routes"][0]
                    dist_km = round(primary_route["distance"] / 1000.0, 2)
                    duration_min = round(primary_route["duration"] / 60.0, 1)
                    coords = primary_route["geometry"]["coordinates"] # list of [lon, lat]
                    
                    route_payload = {
                        "status": "success",
                        "source": "osrm_live",
                        "origin": {
                            "id": origin["Location_ID"],
                            "name": origin["Location_Name"],
                            "lat": origin["Latitude"],
                            "lon": origin["Longitude"],
                            "road_type": origin["Road_Type"],
                            "capacity": origin["Road_Capacity"]
                        },
                        "destination": {
                            "id": dest["Location_ID"],
                            "name": dest["Location_Name"],
                            "lat": dest["Latitude"],
                            "lon": dest["Longitude"],
                            "road_type": dest["Road_Type"],
                            "capacity": dest["Road_Capacity"]
                        },
                        "distance_km": dist_km,
                        "duration_min": duration_min,
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coords
                        }
                    }
                    # Save to cache
                    self.cache[cache_key] = route_payload
                    self._save_cache()
                    return route_payload
                else:
                    return {
                        "status": "error",
                        "message": f"OSRM route calculation error: {data.get('code')}"
                    }
            else:
                return {
                    "status": "error",
                    "message": f"OSRM returned HTTP status {resp.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to connect to road routing service: {str(e)}"
            }

    def get_route_by_coords(
        self,
        origin_lat: float,
        origin_lon: float,
        dest_lat: float,
        dest_lon: float,
        origin_name: Optional[str] = None,
        dest_name: Optional[str] = None,
    ) -> dict:
        """
        Road route between arbitrary geocoded coordinates using the same OSRM
        driving-profile integration as get_route(). Reuses the identical call
        pattern and response shape — routing and estimated routing duration only;
        this endpoint provides no traffic/congestion data.
        Coordinates must come from the geocoder (no fabricated coordinates).
        """
        # Basic sanity: reject identical points and out-of-range values
        if (abs(origin_lat) > 90 or abs(origin_lon) > 180 or
                abs(dest_lat) > 90 or abs(dest_lon) > 180):
            return {"status": "error", "message": "Coordinates out of range"}
        if abs(origin_lat - dest_lat) < 1e-6 and abs(origin_lon - dest_lon) < 1e-6:
            return {"status": "error", "message": "Origin and destination coordinates are identical"}

        cache_key = f"coords:{round(origin_lat, 5)},{round(origin_lon, 5)}->{round(dest_lat, 5)},{round(dest_lon, 5)}"
        if cache_key in self.cache:
            res = self.cache[cache_key].copy()
            res["source"] = "local_cache"
            return res

        url = f"{OSRM_BASE_URL}/{origin_lon},{origin_lat};{dest_lon},{dest_lat}?overview=full&geometries=geojson"
        headers = {
            "User-Agent": "TRAFFIQ-Academic-SmartCity/1.0 (academic; smart-city-analytics)"
        }
        try:
            resp = requests.get(url, headers=headers, timeout=12)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("code") == "Ok" and data.get("routes"):
                    primary_route = data["routes"][0]
                    dist_km = round(primary_route["distance"] / 1000.0, 2)
                    duration_min = round(primary_route["duration"] / 60.0, 1)
                    coords = primary_route["geometry"]["coordinates"]
                    route_payload = {
                        "status": "success",
                        "source": "osrm_live",
                        "origin": {
                            "id": None,
                            "name": origin_name or f"{origin_lat:.5f}, {origin_lon:.5f}",
                            "lat": origin_lat,
                            "lon": origin_lon,
                            "road_type": None,
                            "capacity": None
                        },
                        "freeflow_route": True,
                        "destination": {
                            "id": None, "name": dest_name or f"{dest_lat:.5f}, {dest_lon:.5f}",
                            "lat": dest_lat, "lon": dest_lon,
                            "road_type": None, "capacity": None
                        },
                        "distance_km": dist_km,
                        "duration_min": duration_min,
                        "geometry": {
                            "type": "LineString",
                            "coordinates": coords
                        },
                    }
                    # Save to cache (same cache file as dataset routes)
                    self.cache[cache_key] = route_payload
                    self._save_cache()
                    return route_payload
                else:
                    return {
                        "status": "error",
                        "message": f"OSRM route calculation error: {data.get('code')}"
                    }
            else:
                return {
                    "status": "error",
                    "message": f"OSRM returned HTTP status {resp.status_code}"
                }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to connect to road routing service: {str(e)}"
            }

geospatial_service = GeospatialService()
