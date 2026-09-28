import pandas as pd
import numpy as np
import json
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPTS_DIR.parent / "data"

def inspect_dataset():
    df = pd.read_csv(str(_DATA_DIR / "traffic_raw.csv"))
    
    print("=" * 60)
    print("TRAFFIQ DATASET PROGRAMMATIC INSPECTION REPORT")
    print("=" * 60)
    
    # 1. Row count & column count
    n_rows, n_cols = df.shape
    print(f"\n1. DIMENSIONS:")
    print(f"   Rows: {n_rows:,}")
    print(f"   Columns: {n_cols}")
    
    # 2. Data Types & Missing Values
    print(f"\n2. COLUMNS, DATA TYPES & MISSING VALUES:")
    missing = df.isnull().sum()
    dtypes = df.dtypes
    for col in df.columns:
        print(f"   - {col:<30}: dtype={str(dtypes[col]):<10} missing={missing[col]:<5} ({missing[col]/n_rows*100:.2f}%)")
        
    # 3. Duplicate rows
    dup_rows = df.duplicated().sum()
    print(f"\n3. DUPLICATE ROWS:")
    print(f"   Total exact duplicate rows: {dup_rows}")
    
    # 4. Timestamp analysis
    print(f"\n4. TIMESTAMP ANALYSIS:")
    if "Timestamp" in df.columns:
        df["_parsed_ts"] = pd.to_datetime(df["Timestamp"], errors="coerce")
        min_ts = df["_parsed_ts"].min()
        max_ts = df["_parsed_ts"].max()
        nat_count = df["_parsed_ts"].isna().sum()
        print(f"   Min Timestamp: {min_ts}")
        print(f"   Max Timestamp: {max_ts}")
        print(f"   Duration / Span: {max_ts - min_ts}")
        print(f"   Invalid / NaT Timestamps: {nat_count}")
    
    # 5. Location analysis
    print(f"\n5. UNIQUE LOCATIONS:")
    if "Location" in df.columns:
        locs = df["Location"].value_counts()
        print(f"   Total Unique Locations: {len(locs)}")
        print(f"   Location Distribution:")
        for loc, count in locs.items():
            print(f"     * {loc:<35}: {count:,} records ({count/n_rows*100:.1f}%)")
            
    # 6. Target Variable: Congestion_Level
    print(f"\n6. TARGET VARIABLE (Congestion_Level) ANALYSIS:")
    if "Congestion_Level" in df.columns:
        cl_counts = df["Congestion_Level"].value_counts(dropna=False)
        cl_pcts = df["Congestion_Level"].value_counts(normalize=True, dropna=False) * 100
        print(f"   Classes Present:")
        for cls, count in cl_counts.items():
            print(f"     * {cls:<15}: {count:,} ({cl_pcts[cls]:.2f}%)")
        
        has_low = "Low" in cl_counts
        has_med = "Medium" in cl_counts
        has_high = "High" in cl_counts
        has_crit = "Critical" in cl_counts
        print(f"   Includes Low? {has_low} | Medium? {has_med} | High? {has_high} | Critical? {has_crit}")
        
    # 7. Categorical columns breakdown
    print(f"\n7. CATEGORICAL COLUMNS BREAKDOWN:")
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if "_parsed_ts" in cat_cols: cat_cols.remove("_parsed_ts")
    for col in cat_cols:
        if col in ["Timestamp", "Location"]: continue
        vals = df[col].value_counts(dropna=False).to_dict()
        print(f"   Column '{col}': {vals}")
        
    # 8. Numerical columns summary
    print(f"\n8. NUMERICAL RANGES SUMMARY:")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[num_cols].describe().T[["min", "mean", "50%", "max", "std"]]
    print(desc.to_string())
    
    # 9. Target Leakage & Determination Check
    print(f"\n9. TARGET LEAKAGE & DETERMINATION ANALYSIS:")
    # Check relationship between Congestion_Level and Traffic_to_Capacity_Ratio, Traffic_Volume_veh_hr, Average_Speed_kmph
    if "Congestion_Level" in df.columns:
        print("\n   Grouping by Congestion_Level (mean / min / max of key metrics):")
        metrics_to_check = [c for c in ["Traffic_to_Capacity_Ratio", "Traffic_Volume_veh_hr", "Average_Speed_kmph", "Road_Capacity_veh_hr"] if c in df.columns]
        grouped = df.groupby("Congestion_Level")[metrics_to_check].agg(["min", "median", "mean", "max"])
        print(grouped.to_string())
        
        # Check if Congestion_Level was assigned via direct deterministic thresholding on Traffic_to_Capacity_Ratio
        if "Traffic_to_Capacity_Ratio" in df.columns:
            print("\n   Checking if Congestion_Level is deterministic by T/C Ratio ranges:")
            for cl in df["Congestion_Level"].unique():
                sub = df[df["Congestion_Level"] == cl]["Traffic_to_Capacity_Ratio"]
                print(f"     {cl:<10} -> T/C min: {sub.min():.4f}, max: {sub.max():.4f}")
                
        # Check if Congestion_Level vs Average_Speed_kmph has overlaps
        if "Average_Speed_kmph" in df.columns:
            print("\n   Checking Congestion_Level vs Average_Speed_kmph ranges:")
            for cl in df["Congestion_Level"].unique():
                sub = df[df["Congestion_Level"] == cl]["Average_Speed_kmph"]
                print(f"     {cl:<10} -> Speed min: {sub.min():.2f}, max: {sub.max():.2f}")
                
    # 10. Correlations with numerical variables
    print(f"\n10. CORRELATION WITH CONGESTION LEVEL & KEY NUMERICAL FEATURES:")
    if "Congestion_Level" in df.columns:
        # Create an ordinal encoding to compute correlation
        cl_order = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
        if set(df["Congestion_Level"].unique()).issubset(set(cl_order.keys())):
            df["_cl_code"] = df["Congestion_Level"].map(cl_order)
            corrs = df[num_cols + ["_cl_code"]].corr()["_cl_code"].sort_values(ascending=False)
            print("   Correlation with Congestion_Level (ordinal 0-3):")
            for col, val in corrs.items():
                if col != "_cl_code":
                    print(f"     {col:<30}: {val:.4f}")
        else:
            print(f"   Congestion_Level values differ from standard: {df['Congestion_Level'].unique()}")

if __name__ == "__main__":
    inspect_dataset()
