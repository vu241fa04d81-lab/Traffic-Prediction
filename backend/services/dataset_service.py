"""
TRAFFIQ Dataset Service
Automatic dataset inspection and analytics generation over traffic CSV data.

Data honesty:
- All analytics are computed from the supplied CSV (historical/static data by default).
- Nothing is fabricated: every analytics section is availability-gated and reports
  clearly which required columns were missing instead of inventing values.
- Custom CSV uploads are analyzed in-memory; the default 50,000-row dataset stays
  the default/demo dataset for map, prediction, route and simulation features.
"""

import os
import json
import time
import uuid
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

_SERVICE_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SERVICE_DIR.parent
UPLOADS_DIR = str(_BACKEND_DIR / "data" / "uploads")
GEOCODE_CACHE_PATH = str(_BACKEND_DIR / "data" / "cache" / "geocode_cache.json")
MAX_UPLOAD_BYTES = 60 * 1024 * 1024          # 60 MB
MAX_UPLOAD_ROWS = 200_000                     # analysis cap for uploaded files
MAX_PREVIEW_ROWS = 200
MAX_GEOCODE_UNIQUE_LOCATIONS = 30

# Logical column roles -> accepted column names (first match wins).
COLUMN_ALIASES: Dict[str, List[str]] = {
    "timestamp":  ["Timestamp", "Date_Time", "DateTime", "Datetime", "Date"],
    "location":   ["Location", "Location_Name", "Site", "Area", "Corridor"],
    "location_id":["Location_ID"],
    "congestion": ["Congestion_Level", "Congestion", "Traffic_Condition", "Level", "Congestion_Status"],
    "volume":     ["Traffic_Volume_veh_hr", "Traffic_Volume", "Vehicle_Count", "Volume", "Volume_veh_hr"],
    "speed":      ["Average_Speed_kmph", "Avg_Speed_kmph", "Average_Speed", "Speed_kmph", "Speed"],
    "tcr":        ["Traffic_to_Capacity_Ratio", "Traffic_to_Capacity", "Volume_to_Capacity", "V/C_Ratio"],
    "capacity":   ["Road_Capacity_veh_hr", "Road_Capacity", "Capacity_veh_hr", "Capacity"],
    "hour":       ["Hour", "Hour_of_Day"],
    "dayofweek":  ["DayOfWeek", "Day_Of_Week", "Weekday"],
    "day_name":   ["Day_Name", "Day"],
    "is_weekend": ["Is_Weekend", "Weekend"],
    "peak_hour":  ["Peak_Hour"],
    "weather":    ["Weather", "Weather_Condition"],
    "road_condition": ["Road_Condition", "Road_Surface", "Surface_Condition"],
    "road_type":  ["Road_Type"],
    "accident":   ["Accident_Reported", "Accident", "Incident_Reported"],
    "event":      ["Local_Event", "Event"],
    "holiday":    ["Is_Holiday", "Holiday"],
    "temperature":["Temperature", "Temp_C", "Temperature_C"],
    "visibility": ["Visibility_km", "Visibility"],
    "rainfall":   ["Rainfall_mm", "Rainfall", "Precipitation_mm"],
    "aqi":        ["AQI_Level", "AQI", "Air_Quality_Index"],
    "lanes":      ["Lanes", "Lane_Count"],
    "prev_traffic": ["Previous_Hour_Traffic"],
    "rolling_traffic": ["Rolling_3_Hour_Avg_Traffic"],
    "traffic_change": ["Traffic_Change_Percent"],
    "latitude":   ["Latitude", "Lat"],
    "longitude":  ["Longitude", "Lon", "Lng"],
}

# Features the leakage-free ML pipeline requires at inference time.
ML_REQUIRED_FEATURES = [
    "Location", "Weather", "Road_Condition", "Road_Type", "Temperature",
    "Visibility_km", "Local_Event", "Is_Holiday", "Accident_Reported",
    "AQI_Level", "Hour", "DayOfWeek", "Month", "Lanes", "Road_Capacity_veh_hr",
    "Rainfall_mm", "Is_Weekend", "Peak_Hour", "Previous_Hour_Traffic",
    "Rolling_3_Hour_Avg_Traffic", "Traffic_Change_Percent"
]

# Features derivable from other columns (not required verbatim in a custom CSV)
ML_DERIVABLE = {
    "Is_Weekend": "derivable from DayOfWeek (Sat/Sun) or Timestamp",
    "Peak_Hour": "derivable from Hour (8-10, 17-19 rush windows)",
    "Month": "derivable from Timestamp",
    "Road_Capacity_veh_hr": "derivable per location when Location matches a verified dataset location",
    "Lanes": "derivable per location when Location matches a verified dataset location",
    "Road_Type": "derivable per location when Location matches a verified dataset location",
}

LEVEL_ORDER = ["Low", "Medium", "High", "Critical"]
LEVEL_COLORS = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444", "Critical": "#9333ea"}


def _r(x, n: int = 3):
    """Round + convert numpy scalars to plain python for JSON."""
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if pd.isna(x):
            return None
        return round(float(x), n)
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    return x


def _resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Map each logical role to an actual column in the dataframe (or None)."""
    cols = {c.strip().lower(): c for c in df.columns}
    resolved: Dict[str, Optional[str]] = {}
    for role, candidates in COLUMN_ALIASES.items():
        found = None
        for cand in candidates:
            if cand.lower() in cols:
                found = cols[cand.lower()]
                break
        resolved[role] = found
    return resolved


def _parse_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _level_list(series: pd.Series) -> List[str]:
    """Ordered unique congestion levels present in data (canonical order first)."""
    vals = [str(v) for v in series.dropna().unique()]
    ordered = [lv for lv in LEVEL_ORDER if any(v.lower() == lv.lower() for v in vals)]
    extras = sorted(v for v in vals if v.lower() not in {lv.lower() for lv in ordered})
    return ordered + extras


def _high_share(sub: pd.DataFrame, cong_col: str) -> float:
    if len(sub) == 0:
        return 0.0
    levels = sub[cong_col].astype(str).str.lower()
    return float((levels == "high").mean() * 100.0)


# ------------------------------------------------------------------
# Schema overview
# ------------------------------------------------------------------

def analyze_schema(df: pd.DataFrame, source_label: str) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    missing_total = int(df.isna().sum().sum())

    columns = []
    for col in df.columns:
        s = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(s)
        n_unique = int(s.nunique(dropna=True))
        sample = s.dropna().head(1)
        columns.append({
            "name": col,
            "dtype": ("number" if is_numeric else "text"),
            "missing": int(s.isna().sum()),
            "unique": n_unique,
            "sample": (str(sample.iloc[0])[:40] if len(sample) > 0 else None),
        })

    # Categorical value breakdown (text columns with a manageable cardinality)
    categorical_values = []
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        n_unique = int(s.nunique(dropna=True))
        if 0 < n_unique <= 25:
            vc = s.value_counts(dropna=False).head(8)
            categorical_values.append({
                "column": col,
                "unique": n_unique,
                "top_values": [{"value": ("" if pd.isna(k) else str(k)), "count": _r(v)} for k, v in vc.items()],
            })

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    numeric_summary = []
    if numeric_cols:
        desc = df[numeric_cols].describe().T
        for col in numeric_cols:
            numeric_summary.append({
                "column": col,
                "mean": _r(desc.loc[col, "mean"], 2),
                "std": _r(desc.loc[col, "std"], 2),
                "min": _r(desc.loc[col, "min"], 2),
                "median": _r(desc.loc[col, "50%"], 2),
                "max": _r(desc.loc[col, "max"], 2),
            })

    # Timestamp range
    ts_range: Dict[str, Any] = {"available": False}
    ts_col = _resolve_columns(df)["timestamp"]
    if ts_col is not None:
        parsed = _parse_timestamps(df[ts_col])
        valid = parsed.dropna()
        if len(valid) > 0:
            ts_range = {
                "available": True,
                "column": ts_col,
                "start": str(valid.min()),
                "end": str(valid.max()),
                "invalid_or_missing": int(parsed.isna().sum()),
            }

    loc_col = _resolve_columns(df)["location"]
    unique_locations = int(df[loc_col].nunique(dropna=True)) if loc_col else 0

    return {
        "source": source_label,
        "row_count": int(n_rows),
        "column_count": int(n_cols),
        "missing_cells": missing_total,
        "duplicate_rows": int(df.duplicated().sum()),
        "unique_locations": unique_locations,
        "location_column": loc_col,
        "timestamp_range": ts_range,
        "columns": columns,
        "categorical_values": categorical_values,
        "numeric_summary": numeric_summary,
    }


# ------------------------------------------------------------------
# Automatic analytics (each section availability-gated)
# ------------------------------------------------------------------

def compute_full_analytics(df: pd.DataFrame, resolved: Dict[str, Optional[str]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    cong = resolved["congestion"]
    vol = resolved["volume"]
    spd = resolved["speed"]
    tcr = resolved["tcr"]
    loc = resolved["location"]
    hour = resolved["hour"]
    weather = resolved["weather"]
    road = resolved["road_condition"]

    # ---- 1. Congestion Level Distribution ----
    if cong:
        vc = df[cong].value_counts()
        levels = _level_list(df[cong])
        total = int(vc.sum())
        out["congestion_distribution"] = {
            "available": True,
            "levels": levels,
            "colors": {lv: LEVEL_COLORS.get(lv, "#64748b") for lv in levels},
            "counts": {lv: _r(vc.get(lv, 0)) for lv in levels},
            "percentages": {lv: _r(vc.get(lv, 0) / total * 100, 2) if total else 0 for lv in levels},
        }
    else:
        out["congestion_distribution"] = {"available": False, "reason": "No congestion level column found (looked for Congestion_Level / Congestion / Traffic_Condition)."}

    # ---- 2. Traffic Volume by Hour ----
    if vol and hour:
        grp = df.groupby(hour)[vol].agg(["mean", "max"]).reset_index()
        out["volume_by_hour"] = {
            "available": True,
            "hours": [_r(h) for h in grp[hour]],
            "avg_volume": [_r(v, 1) for v in grp["mean"]],
            "max_volume": [_r(v) for v in grp["max"]],
        }
    else:
        out["volume_by_hour"] = {"available": False, "reason": "Requires a traffic volume column and an hour column."}

    # ---- 3. Congestion by Hour (stacked shares) ----
    if cong and hour:
        levels = _level_list(df[cong])
        pivot = df.groupby(hour)[cong].value_counts(normalize=True).unstack(fill_value=0) * 100
        out["congestion_by_hour"] = {
            "available": True,
            "levels": levels,
            "hours": [int(h) for h in pivot.index],
            "shares": {lv: [_r(pivot[lv].get(h, 0.0), 2) if lv in pivot.columns else 0.0 for h in pivot.index] for lv in levels},
            "avg_tcr_by_hour": ( [_r(v, 3) for v in df.groupby(hour)[tcr].mean()] if tcr else None ),
        }
    else:
        out["congestion_by_hour"] = {"available": False, "reason": "Requires a congestion level column and an hour column."}

    # ---- 4. Location-wise Congestion ----
    if cong and loc:
        levels = _level_list(df[cong])
        rows = []
        for lname, sub in df.groupby(loc):
            n = len(sub)
            row: Dict[str, Any] = {"location": str(lname), "records": int(n)}
            for lv in levels:
                row[f"{lv.lower()}_pct"] = _r((sub[cong].astype(str).str.lower() == lv.lower()).mean() * 100, 2)
            if vol: row["avg_volume"] = _r(sub[vol].mean(), 1)
            if spd: row["avg_speed"] = _r(sub[spd].mean(), 1)
            if tcr: row["avg_tcr"] = _r(sub[tcr].mean(), 3)
            rows.append(row)
        rows.sort(key=lambda r: r.get("high_pct", 0), reverse=True)
        out["location_congestion"] = {"available": True, "levels": levels, "locations": rows}
    else:
        out["location_congestion"] = {"available": False, "reason": "Requires congestion level and location columns."}

    # ---- 5. Weather vs Congestion ----
    if cong and weather:
        rows = []
        for w, sub in df.groupby(weather):
            rows.append({
                "weather": str(w),
                "records": int(len(sub)),
                "high_congestion_pct": _r(_high_share(sub, cong), 2),
                "avg_speed": _r(sub[spd].mean(), 1) if spd else None,
                "avg_volume": _r(sub[vol].mean(), 1) if vol else None,
                "avg_tcr": _r(sub[tcr].mean(), 3) if tcr else None,
            })
        rows.sort(key=lambda r: r["high_congestion_pct"], reverse=True)
        out["weather_congestion"] = {"available": True, "rows": rows}
    else:
        out["weather_congestion"] = {"available": False, "reason": "Requires congestion level and weather columns."}

    # ---- 6. Road Condition vs Congestion ----
    if cong and road:
        rows = []
        for rc, sub in df.groupby(road):
            rows.append({
                "road_condition": str(rc),
                "records": int(len(sub)),
                "high_congestion_pct": _r(_high_share(sub, cong), 2),
                "avg_speed": _r(sub[spd].mean(), 1) if spd else None,
                "avg_tcr": _r(sub[tcr].mean(), 3) if tcr else None,
            })
        rows.sort(key=lambda r: r["high_congestion_pct"], reverse=True)
        out["road_condition_congestion"] = {"available": True, "rows": rows}
    else:
        out["road_condition_congestion"] = {"available": False, "reason": "Requires congestion level and road condition columns."}

    # ---- 7. Weekday vs Weekend ----
    iswe = resolved["is_weekend"]
    dow = resolved["dayofweek"]
    weekend_series: Optional[pd.Series] = None
    weekend_source = None
    if iswe:
        weekend_series = pd.to_numeric(df[iswe], errors="coerce").fillna(0).astype(int)
        weekend_source = iswe
    elif dow:
        weekend_series = pd.to_numeric(df[dow], errors="coerce").isin([5, 6]).astype(int)
        weekend_source = f"{dow} (Sat/Sun)"
    elif resolved["timestamp"]:
        parsed = _parse_timestamps(df[resolved["timestamp"]])
        weekend_series = parsed.dt.dayofweek.isin([5, 6]).astype(int)
        weekend_source = "Timestamp day-of-week"
    if cong and weekend_series is not None:
        tmp = df[[cong] + ([vol] if vol else []) + ([spd] if spd else [])].copy()
        tmp["_wknd"] = weekend_series.values
        rows = []
        for flag, label in [(0, "Weekday"), (1, "Weekend")]:
            sub = tmp[tmp["_wknd"] == flag]
            row: Dict[str, Any] = {
                "segment": label,
                "records": int(len(sub)),
                "high_congestion_pct": _r(_high_share(sub, cong), 2),
                "medium_congestion_pct": _r((sub[cong].astype(str).str.lower() == "medium").mean() * 100, 2),
                "low_congestion_pct": _r((sub[cong].astype(str).str.lower() == "low").mean() * 100, 2),
            }
            if vol: row["avg_volume"] = _r(sub[vol].mean(), 1)
            if spd: row["avg_speed"] = _r(sub[spd].mean(), 1)
            rows.append(row)
        out["weekday_weekend"] = {"available": True, "source": weekend_source, "rows": rows}
    else:
        out["weekday_weekend"] = {"available": False, "reason": "Requires congestion level plus Is_Weekend / DayOfWeek / Timestamp."}

    # ---- 8. Accident vs Congestion ----
    if cong and resolved["accident"]:
        acc = resolved["accident"]
        rows = []
        for flag, label in [(0, "No accident reported"), (1, "Accident reported")]:
            sub = df[pd.to_numeric(df[acc], errors="coerce").fillna(0).astype(int) == flag]
            rows.append({
                "segment": label,
                "records": int(len(sub)),
                "high_congestion_pct": _r(_high_share(sub, cong), 2),
                "avg_speed": _r(sub[spd].mean(), 1) if spd else None,
                "avg_volume": _r(sub[vol].mean(), 1) if vol else None,
            })
        out["accident_congestion"] = {"available": True, "rows": rows}
    else:
        out["accident_congestion"] = {"available": False, "reason": "Requires congestion level and accident columns."}

    # ---- 9. Local Event vs Congestion ----
    if cong and resolved["event"]:
        ev = resolved["event"]
        rows = []
        for flag, label in [(0, "No local event"), (1, "Local event active")]:
            sub = df[pd.to_numeric(df[ev], errors="coerce").fillna(0).astype(int) == flag]
            rows.append({
                "segment": label,
                "records": int(len(sub)),
                "high_congestion_pct": _r(_high_share(sub, cong), 2),
                "avg_speed": _r(sub[spd].mean(), 1) if spd else None,
                "avg_volume": _r(sub[vol].mean(), 1) if vol else None,
            })
        out["event_congestion"] = {"available": True, "rows": rows}
    else:
        out["event_congestion"] = {"available": False, "reason": "Requires congestion level and local event columns."}

    # ---- 10. Hour x Location Congestion Heatmap ----
    if cong and loc and hour:
        levels = _level_list(df[cong])
        locations = sorted([str(l) for l in df[loc].dropna().unique()])
        piv = df.copy()
        piv["_is_high"] = (piv[cong].astype(str).str.lower() == "high").astype(int)
        high_m = piv.pivot_table(index=hour, columns=loc, values="_is_high", aggfunc="mean") * 100
        tcr_m = df.pivot_table(index=hour, columns=loc, values=tcr, aggfunc="mean") if tcr else None
        out["hour_location_heatmap"] = {
            "available": True,
            "hours": [int(h) for h in sorted(df[hour].dropna().unique())],
            "locations": locations,
            "high_pct": {str(h): {l: _r(high_m.loc[h, l], 2) if (h in high_m.index and l in high_m.columns and pd.notna(high_m.loc[h, l])) else None for l in locations} for h in high_m.index},
            "avg_tcr": ({str(h): {l: _r(tcr_m.loc[h, l], 3) if (h in tcr_m.index and l in tcr_m.columns and pd.notna(tcr_m.loc[h, l])) else None for l in locations} for h in tcr_m.index} if tcr_m is not None else None),
            "levels": levels,
        }
    else:
        out["hour_location_heatmap"] = {"available": False, "reason": "Requires congestion level, location and hour columns."}

    # ---- 11/12. Volume & Speed trends over time ----
    ts_col = resolved["timestamp"]
    if ts_col:
        parsed = _parse_timestamps(df[ts_col])
        tdf = df[[c for c in [vol, spd] if c]].copy()
        tdf["_date"] = parsed.dt.date
        tdf = tdf.dropna(subset=["_date"])
        daily = tdf.groupby("_date").mean(numeric_only=True).reset_index().sort_values("_date")
        out["volume_trend"] = {
            "available": vol is not None,
            "granularity": "daily (historical averages)",
            "dates": [str(d) for d in daily["_date"]][:400],
            "values": ([_r(v, 1) for v in daily[vol]][:400] if vol else None),
            "reason": None if vol else "No traffic volume column found.",
        }
        out["speed_trend"] = {
            "available": spd is not None,
            "granularity": "daily (historical averages)",
            "dates": [str(d) for d in daily["_date"]][:400],
            "values": ([_r(v, 1) for v in daily[spd]][:400] if spd else None),
            "reason": None if spd else "No average speed column found.",
        }
    else:
        for key, col in [("volume_trend", vol), ("speed_trend", spd)]:
            if col and hour:
                grp = df.groupby(hour)[col].mean().reset_index()
                out[key] = {"available": True, "granularity": "hourly (no timestamp column)", "dates": [f"{int(h):02d}:00" for h in grp[hour]], "values": [_r(v, 1) for v in grp[col]], "reason": None}
            else:
                out[key] = {"available": False, "reason": "Requires a timestamp column (or hour + metric column)."}

    # ---- 13. Traffic-to-Capacity analysis ----
    if tcr:
        s = pd.to_numeric(df[tcr], errors="coerce").dropna()
        bins = [i / 10 for i in range(0, 11)]
        hist = pd.cut(s, bins=[-0.001] + bins, right=True).value_counts().sort_index()
        labels = ["<0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0", ">1.0"]
        out["tcr_analysis"] = {
            "available": True,
            "histogram_labels": labels[:len(hist)],
            "histogram_counts": [_r(v) for v in hist.values],
            "avg_tcr": _r(s.mean(), 3),
            "pct_above_0_7": _r((s > 0.7).mean() * 100, 2),
            "pct_above_0_85": _r((s > 0.85).mean() * 100, 2),
            "avg_tcr_by_hour": ([_r(v, 3) for v in df.groupby(hour)[tcr].mean()] if hour else None),
        }
    else:
        out["tcr_analysis"] = {"available": False, "reason": "No traffic-to-capacity ratio column found (also not derivable without volume + capacity)."}

    # ---- 14. Corridor comparison (ranked) ----
    if cong and loc:
        loc_rows = out["location_congestion"]["locations"] if "location_congestion" in out and out["location_congestion"].get("available") else []
        ranked = sorted(loc_rows, key=lambda r: r.get("high_pct", 0), reverse=True)
        out["corridor_comparison"] = {
            "available": len(ranked) > 0,
            "highest_risk": ranked[0] if ranked else None,
            "lowest_risk": ranked[-1] if ranked else None,
            "ranking": ranked,
        }
    else:
        out["corridor_comparison"] = {"available": False, "reason": "Requires congestion level and location columns."}

    # ---- 15. Numeric correlations with congestion (ordinal) ----
    if cong:
        level_order_map = {lv.lower(): i for i, lv in enumerate(LEVEL_ORDER)}
        codes = df[cong].astype(str).str.lower().map(level_order_map)
        if codes.notna().all() and len(df.select_dtypes(include=[np.number]).columns) > 0:
            tmp = df.select_dtypes(include=[np.number]).copy()
            tmp["_cong_code"] = codes
            corr = tmp.corr(numeric_only=True)["_cong_code"].drop("_cong_code").dropna()
            corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(8)
            out["numeric_correlations"] = {
                "available": True,
                "note": "Pearson correlation of numeric columns with ordinal congestion level (Low=0 ... Critical=3). Correlation is not causation.",
                "rows": [{"column": c, "correlation": _r(v, 3)} for c, v in corr.items()],
            }
        else:
            out["numeric_correlations"] = {"available": False, "reason": "Congestion levels outside the standard Low/Medium/High/Critical scale."}
    else:
        out["numeric_correlations"] = {"available": False, "reason": "Requires a congestion level column."}

    return out


# ------------------------------------------------------------------
# Preview / search
# ------------------------------------------------------------------

def build_preview(df: pd.DataFrame, search: Optional[str], limit: int) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 20), MAX_PREVIEW_ROWS))
    d = df
    matched_total: Optional[int] = None
    if search:
        q = str(search).strip().lower()
        str_cols = [c for c in d.columns if not pd.api.types.is_numeric_dtype(d[c])]
        if str_cols and q:
            mask = None
            for c in str_cols:
                m = d[c].astype(str).str.lower().str.contains(q, regex=False, na=False)
                mask = m if mask is None else (mask | m)
            d = d[mask]
            matched_total = int(len(d))
    total = len(d)
    head = d.head(limit)
    cols = list(head.columns)
    rows = [[("" if pd.isna(v) else (v if isinstance(v, (int, float, bool)) and not isinstance(v, (np.integer, np.floating, np.bool_)) else str(v))) for v in row] for row in head.itertuples(index=False, name=None)]
    # normalize numpy scalars
    rows = [[_r(v) if isinstance(v, (np.integer, np.floating, np.bool_)) else v for v in row] for row in rows]
    return {
        "columns": cols,
        "rows": rows,
        "returned": len(rows),
        "matched_total": matched_total if matched_total is not None else total,
        "dataset_total": int(len(df)),
        "truncated": total > len(rows),
    }


# ------------------------------------------------------------------
# Custom upload analysis. The default dataset's overview/analytics/preview
# are served by analytics_service — the single shared data pipeline already
# used by the Historical Map (see main.py and backend/services/analytics_service.py).
# ------------------------------------------------------------------

class DatasetService:
    def __init__(self):
        self._geocode_cache: Optional[Dict[str, Any]] = None

    # ---- geocoding (Nominatim, cached) for custom CSVs ----
    def _load_geocode_cache(self) -> Dict[str, Any]:
        if self._geocode_cache is None:
            if os.path.exists(GEOCODE_CACHE_PATH):
                try:
                    with open(GEOCODE_CACHE_PATH, "r", encoding="utf-8") as f:
                        self._geocode_cache = json.load(f)
                except Exception:
                    self._geocode_cache = {}
            else:
                self._geocode_cache = {}
        return self._geocode_cache

    def _persist_geocode_cache(self):
        try:
            os.makedirs(os.path.dirname(GEOCODE_CACHE_PATH), exist_ok=True)
            with open(GEOCODE_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(self._geocode_cache, f, indent=2)
        except Exception as e:
            print(f"Warning: could not persist geocode cache: {e}")

    def geocode_locations(self, names: List[str]) -> Dict[str, Any]:
        """Geocode unique location names via OpenStreetMap Nominatim (cached, rate-limited)."""
        try:
            from backend.services.geospatial_service import geocode_location
        except ImportError:
            from services.geospatial_service import geocode_location
        cache = self._load_geocode_cache()
        results = []
        geocoded_count = 0
        for name in names:
            name = str(name).strip()
            if not name:
                continue
            if name in cache:
                entry = cache[name]
            else:
                entry = geocode_location(name)
                cache[name] = entry
                self._persist_geocode_cache()
                time.sleep(1.1)  # Nominatim usage policy: max 1 request/second
            results.append({
                "location": name,
                "lat": entry.get("lat"),
                "lon": entry.get("lon"),
                "display_name": entry.get("display_name"),
                "status": entry.get("status", "error"),
            })
            if entry.get("status") == "ok":
                geocoded_count += 1
        return {"attempted": len(results), "geocoded": geocoded_count, "results": results}

    def geocode_single(self, name: str) -> Dict[str, Any]:
        """Geocode one place name via the existing Nominatim geocoder (persistent cache, no fabrication).
        Used by the map / route place search for any real-world location."""
        try:
            from backend.services.geospatial_service import geocode_location
        except ImportError:
            from services.geospatial_service import geocode_location
        name = str(name).strip()
        if not name:
            return {"status": "not_found", "lat": None, "lon": None, "display_name": None}
        cache = self._load_geocode_cache()
        entry = cache.get(name)
        if entry is None:
            entry = geocode_location(name)
            cache[name] = entry
            self._persist_geocode_cache()
        return dict(entry)

    # ---- custom CSV upload analysis ----
    def analyze_upload(self, file_bytes: bytes, filename: str) -> Dict[str, Any]:
        if len(file_bytes) > MAX_UPLOAD_BYTES:
            raise ValueError(f"File too large ({len(file_bytes)/1e6:.1f} MB). Limit is 60 MB.")

        import io
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as e:
            raise ValueError(f"Could not parse CSV: {e}")

        if df.empty:
            raise ValueError("The uploaded CSV contains no data rows.")
        sampled_note = None
        if len(df) > MAX_UPLOAD_ROWS:
            df = df.sample(n=MAX_UPLOAD_ROWS, random_state=42).reset_index(drop=True)
            sampled_note = f"File has more than {MAX_UPLOAD_ROWS:,} rows; analytics computed on a random sample of {MAX_UPLOAD_ROWS:,} rows."

        resolved = _resolve_columns(df)
        overview = analyze_schema(df, f"Custom uploaded CSV: {filename} — historical/static data")
        analytics = compute_full_analytics(df, resolved)

        # ML feature availability for this schema
        present_cols = set(df.columns)
        ml_features = []
        missing_required = []
        for feat in ML_REQUIRED_FEATURES:
            if feat in present_cols:
                status = "present"
            elif feat in ML_DERIVABLE:
                status = "derivable"
            else:
                status = "missing"
                missing_required.append(feat)
            ml_features.append({"feature": feat, "status": status, "note": ML_DERIVABLE.get(feat)})

        # Congestion target availability
        target_available = resolved["congestion"] is not None

        # Geography: use lat/lon if present, else geocode unique location names (bounded)
        geography: Dict[str, Any] = {"available": False, "method": None, "locations": [], "notice": None}
        lat_col, lon_col = resolved["latitude"], resolved["longitude"]
        loc_col = resolved["location"]
        try:
            if lat_col and lon_col:
                pts = df[[lat_col, lon_col]].dropna()
                uniq = pts.drop_duplicates().head(MAX_GEOCODE_UNIQUE_LOCATIONS)
                geography = {
                    "available": True,
                    "method": "latitude/longitude columns used directly (no fabrication)",
                    "locations": [{"location": (str(df.loc[idx, loc_col]) if loc_col else f"Point {i+1}"), "lat": _r(row[lat_col], 6), "lon": _r(row[lon_col], 6), "status": "provided_in_file"} for i, (idx, row) in enumerate(uniq.iterrows())],
                }
            elif loc_col:
                uniq_names = [str(v) for v in df[loc_col].dropna().unique()]
                if 0 < len(uniq_names) <= MAX_GEOCODE_UNIQUE_LOCATIONS:
                    geo = self.geocode_locations(uniq_names)
                    partial_notice = None
                    if 0 < geo["geocoded"] < geo["attempted"]:
                        partial_notice = (f"{geo['geocoded']} of {geo['attempted']} location names were geocoded via OpenStreetMap; "
                                          "unresolvable names are excluded from map visualization rather than approximated.")
                    geography = {
                        "available": geo["geocoded"] > 0,
                        "method": "OpenStreetMap Nominatim geocoding of location names",
                        "locations": geo["results"],
                        "notice": partial_notice if geo["geocoded"] > 0 else "Location names could not be geocoded via OpenStreetMap; no map will be fabricated.",
                    }
                elif len(uniq_names) > MAX_GEOCODE_UNIQUE_LOCATIONS:
                    geography = {
                        "available": False,
                        "method": None,
                        "locations": [],
                        "notice": f"Dataset has {len(uniq_names)} unique location names (limit {MAX_GEOCODE_UNIQUE_LOCATIONS} for automatic geocoding); map visualization was not generated.",
                    }
        except Exception as e:
            geography = {"available": False, "method": None, "locations": [], "notice": f"Geocoding failed: {e}"}

        # Persist upload for provenance (analysis already computed in-memory)
        dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
        try:
            os.makedirs(UPLOADS_DIR, exist_ok=True)
            with open(os.path.join(UPLOADS_DIR, f"{dataset_id}.csv"), "wb") as f:
                f.write(file_bytes)
            with open(os.path.join(UPLOADS_DIR, f"{dataset_id}.json"), "w", encoding="utf-8") as f:
                json.dump({"original_filename": filename, "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"), "rows": int(len(df)), "columns": int(df.shape[1])}, f, indent=2)
        except Exception as e:
            print(f"Warning: could not persist uploaded file: {e}")

        return {
            "status": "success",
            "dataset_id": dataset_id,
            "filename": filename,
            "data_nature": "Historical / static CSV upload — not live or real-time traffic",
            "overview": overview,
            "analytics": analytics,
            "ml_feature_availability": {
                "target_available": target_available,
                "features": ml_features,
                "missing_required": missing_required,
                "notice": self._ml_availability_notice(target_available, missing_required),
            },
            "geography": geography,
            "sampled_note": sampled_note,
        }

    @staticmethod
    def _ml_availability_notice(target_available: bool, missing_required: List[str]) -> str:
        if target_available and not missing_required:
            return ("All ML model features are present or derivable. Note: the deployed TRAFFIQ model was trained on the "
                    "default 50,000-row dataset; predictions for custom datasets would require retraining, which this demo does not do automatically.")
        parts = []
        if not target_available:
            parts.append("No congestion level column was found, so congestion-distribution analytics are unavailable for this file.")
        if missing_required:
            parts.append(f"ML model features missing from this schema: {', '.join(missing_required)}. "
                         "Missing features are not invented; ML prediction for this dataset is therefore not offered.")
        if not parts:
            parts.append("Schema is compatible with the analytics suite.")
        return " ".join(parts)


dataset_service = DatasetService()
