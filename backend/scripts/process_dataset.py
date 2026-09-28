import pandas as pd
import numpy as np
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPTS_DIR.parent / "data"

def process_traffic_data():
    raw_path = str(_DATA_DIR / "traffic_raw.csv")
    processed_path = str(_DATA_DIR / "traffic_processed.csv")
    locations_path = str(_DATA_DIR / "locations.csv")
    
    print("Reading raw dataset...")
    df = pd.read_csv(raw_path)
    df_loc = pd.read_csv(locations_path)
    
    # 1. Parse timestamps and sort
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values(["Location", "Timestamp"]).reset_index(drop=True)
    
    # 2. Extract calendar helpers
    day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    df["Day_Name"] = df["DayOfWeek"].map(day_map)
    df["Date"] = df["Timestamp"].dt.date.astype(str)
    
    # 3. Merge Location_ID from locations master
    loc_id_map = dict(zip(df_loc["Location_Name"], df_loc["Location_ID"]))
    df["Location_ID"] = df["Location"].map(loc_id_map)
    
    # 4. Standardize types
    int_cols = ["Local_Event", "Is_Holiday", "Accident_Reported", "AQI_Level", "Hour", "DayOfWeek", "Month", "Lanes", "Road_Capacity_veh_hr", "Is_Weekend", "Peak_Hour", "Traffic_Volume_veh_hr"]
    for c in int_cols:
        df[c] = df[c].astype(int)
        
    float_cols = ["Temperature", "Visibility_km", "Rainfall_mm", "Traffic_to_Capacity_Ratio", "Average_Speed_kmph", "Previous_Hour_Traffic", "Rolling_3_Hour_Avg_Traffic", "Traffic_Change_Percent"]
    for c in float_cols:
        df[c] = df[c].round(4)
        
    # Reorder columns with Location_ID near Location
    cols = ["Timestamp", "Date", "Location_ID", "Location", "Day_Name"] + [c for c in df.columns if c not in ["Timestamp", "Date", "Location_ID", "Location", "Day_Name"]]
    df = df[cols]
    
    # 5. Save processed CSV
    df.to_csv(processed_path, index=False)
    print(f"Successfully processed {len(df):,} rows and {len(df.columns)} columns into {processed_path}")
    print(f"Sample columns: {df.columns.tolist()[:10]}")
    
if __name__ == "__main__":
    process_traffic_data()
