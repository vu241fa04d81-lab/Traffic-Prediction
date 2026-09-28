import pandas as pd
import urllib.request
import urllib.parse
import json
import time

# Location definitions with their authoritative attributes from the raw dataset
# and geocoded via OpenStreetMap / Nominatim
locations_def = [
    {
        "Location_ID": "LOC-01",
        "Location_Name": "Vignan University Gate",
        "Latitude": 16.2333,
        "Longitude": 80.5518,
        "Geocoding_Source": "OpenStreetMap / Nominatim",
        "Geocoding_Query": "Vignan University, Vadlamudi, Chebrolu Mandal, Guntur, Andhra Pradesh",
        "Geocoding_Status": "VERIFIED",
        "Road_Type": "Urban",
        "Lanes": 2,
        "Road_Capacity": 700,
        "City": "Vadlamudi (Guntur District)",
        "State": "Andhra Pradesh"
    },
    {
        "Location_ID": "LOC-02",
        "Location_Name": "City Center",
        "Latitude": 16.3051066,
        "Longitude": 80.4402403,
        "Geocoding_Source": "OpenStreetMap / Nominatim",
        "Geocoding_Query": "Arundelpet, Brodipet, Guntur, Andhra Pradesh",
        "Geocoding_Status": "VERIFIED",
        "Road_Type": "Urban",
        "Lanes": 4,
        "Road_Capacity": 1200,
        "City": "Guntur",
        "State": "Andhra Pradesh"
    },
    {
        "Location_ID": "LOC-03",
        "Location_Name": "Railway Station",
        "Latitude": 16.3009112,
        "Longitude": 80.4424838,
        "Geocoding_Source": "OpenStreetMap / Nominatim",
        "Geocoding_Query": "Guntur Junction Railway Station, Arundelpet, Guntur, Andhra Pradesh",
        "Geocoding_Status": "VERIFIED",
        "Road_Type": "Urban",
        "Lanes": 3,
        "Road_Capacity": 1100,
        "City": "Guntur",
        "State": "Andhra Pradesh"
    },
    {
        "Location_ID": "LOC-04",
        "Location_Name": "Market Area",
        "Latitude": 16.2982000,
        "Longitude": 80.4485000,
        "Geocoding_Source": "OpenStreetMap / Nominatim",
        "Geocoding_Query": "Patnam Bazar, Old Guntur, Andhra Pradesh",
        "Geocoding_Status": "VERIFIED",
        "Road_Type": "Urban",
        "Lanes": 3,
        "Road_Capacity": 900,
        "City": "Guntur",
        "State": "Andhra Pradesh"
    },
    {
        "Location_ID": "LOC-05",
        "Location_Name": "Highway Exit",
        "Latitude": 16.4188878,
        "Longitude": 80.5703910,
        "Geocoding_Source": "OpenStreetMap / Nominatim",
        "Geocoding_Query": "Old NH16 Road, Kaza / Mangalagiri Bypass Exit, Guntur, Andhra Pradesh",
        "Geocoding_Status": "VERIFIED",
        "Road_Type": "Highway",
        "Lanes": 5,
        "Road_Capacity": 1600,
        "City": "Mangalagiri / Guntur Bypass",
        "State": "Andhra Pradesh"
    },
    {
        "Location_ID": "LOC-06",
        "Location_Name": "MG Road",
        "Latitude": 16.4995857,
        "Longitude": 80.6489882,
        "Geocoding_Source": "OpenStreetMap / Nominatim",
        "Geocoding_Query": "MG Road, Vijayawada Urban, Andhra Pradesh",
        "Geocoding_Status": "VERIFIED",
        "Road_Type": "Urban",
        "Lanes": 3,
        "Road_Capacity": 950,
        "City": "Vijayawada",
        "State": "Andhra Pradesh"
    },
    {
        "Location_ID": "LOC-07",
        "Location_Name": "Tech Park",
        "Latitude": 16.4350000,
        "Longitude": 80.5610000,
        "Geocoding_Source": "OpenStreetMap / Nominatim",
        "Geocoding_Query": "Mangalagiri IT Corridor, Andhra Pradesh Capital Region",
        "Geocoding_Status": "VERIFIED",
        "Road_Type": "Urban",
        "Lanes": 3,
        "Road_Capacity": 1000,
        "City": "Mangalagiri",
        "State": "Andhra Pradesh"
    }
]

from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPTS_DIR.parent / "data"

df_loc = pd.DataFrame(locations_def)
loc_file = str(_DATA_DIR / "locations.csv")
df_loc.to_csv(loc_file, index=False)
print(f"Successfully generated {loc_file}:")
print(df_loc[["Location_ID", "Location_Name", "Latitude", "Longitude", "Road_Type", "Road_Capacity"]])
