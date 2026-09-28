import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPTS_DIR.parent
for _p in [str(_BACKEND_DIR), str(_BACKEND_DIR.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import time
try:
    from backend.services.geospatial_service import geospatial_service
except ImportError:
    from services.geospatial_service import geospatial_service

def precache_all_routes():
    locations = geospatial_service.get_all_locations()
    print(f"Pre-caching OSRM routes between {len(locations)} locations...")
    
    total = len(locations) * (len(locations) - 1)
    count = 0
    success = 0
    
    for origin in locations:
        for dest in locations:
            if origin["Location_ID"] == dest["Location_ID"]:
                continue
            count += 1
            print(f"[{count}/{total}] Fetching {origin['Location_Name']} -> {dest['Location_Name']}...", end=" ")
            res = geospatial_service.get_route(origin["Location_ID"], dest["Location_ID"])
            if res.get("status") == "success":
                success += 1
                print(f"OK ({res['distance_km']} km, {res['duration_min']} min, {len(res['geometry']['coordinates'])} pts)")
            else:
                print(f"FAILED: {res.get('message')}")
            time.sleep(0.3) # gentle rate
            
    print(f"\nPre-cache complete! Successfully cached {success}/{total} routes.")

if __name__ == "__main__":
    precache_all_routes()
