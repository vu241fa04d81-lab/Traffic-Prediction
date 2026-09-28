"""
TRAFFIQ Analytics Service
Provides high-performance Level 1 analytics over processed traffic dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

_SERVICE_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SERVICE_DIR.parent
PROCESSED_DATA_PATH = str(_BACKEND_DIR / "data" / "traffic_processed.csv")
LOCATIONS_PATH = str(_BACKEND_DIR / "data" / "locations.csv")

class AnalyticsService:
    def __init__(self):
        self.df = pd.read_csv(PROCESSED_DATA_PATH)
        self.locations_df = pd.read_csv(LOCATIONS_PATH)
        self._summary_cache = None
        self._overview_cache = None
        self._analytics_cache = None

    def get_summary(self) -> Dict[str, Any]:
        """Calculates global dataset analytics and trend aggregations."""
        if self._summary_cache:
            return self._summary_cache

        df = self.df
        total_records = len(df)
        
        # Congestion distribution
        cl_counts = df["Congestion_Level"].value_counts().to_dict()
        cl_pcts = {k: round(v / total_records * 100, 2) for k, v in cl_counts.items()}
        
        # Weather vs Congestion
        weather_grp = df.groupby("Weather").agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            avg_tc_ratio=("Traffic_to_Capacity_Ratio", "mean"),
            high_congestion_pct=("Congestion_Level", lambda x: (x == "High").mean() * 100),
            records=("Congestion_Level", "count")
        ).round(2).reset_index().to_dict(orient="records")

        # Road Condition vs Congestion
        road_grp = df.groupby("Road_Condition").agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            avg_tc_ratio=("Traffic_to_Capacity_Ratio", "mean"),
            high_congestion_pct=("Congestion_Level", lambda x: (x == "High").mean() * 100),
            records=("Congestion_Level", "count")
        ).round(2).reset_index().to_dict(orient="records")

        # Hourly Volume & Speed
        hourly_grp = df.groupby("Hour").agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            avg_tc_ratio=("Traffic_to_Capacity_Ratio", "mean"),
            high_congestion_pct=("Congestion_Level", lambda x: (x == "High").mean() * 100)
        ).round(2).reset_index().to_dict(orient="records")

        # Day of Week trends
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        dow_grp = df.groupby(["DayOfWeek", "Day_Name"]).agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            high_congestion_pct=("Congestion_Level", lambda x: (x == "High").mean() * 100)
        ).round(2).reset_index()
        # sort by DayOfWeek
        dow_grp = dow_grp.sort_values("DayOfWeek").to_dict(orient="records")

        # Peak Hour vs Off-Peak
        peak_comp = df.groupby("Peak_Hour").agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            avg_tc_ratio=("Traffic_to_Capacity_Ratio", "mean"),
            high_congestion_pct=("Congestion_Level", lambda x: (x == "High").mean() * 100)
        ).round(2).reset_index().to_dict(orient="records")

        # Accident Impact
        accident_comp = df.groupby("Accident_Reported").agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            high_congestion_pct=("Congestion_Level", lambda x: (x == "High").mean() * 100)
        ).round(2).reset_index().to_dict(orient="records")

        # Local Event Impact
        event_comp = df.groupby("Local_Event").agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            high_congestion_pct=("Congestion_Level", lambda x: (x == "High").mean() * 100)
        ).round(2).reset_index().to_dict(orient="records")

        # Location comparison summary table
        loc_comp = df.groupby("Location").agg(
            records=("Congestion_Level", "count"),
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            avg_tc_ratio=("Traffic_to_Capacity_Ratio", "mean"),
            max_volume=("Traffic_Volume_veh_hr", "max"),
            min_speed=("Average_Speed_kmph", "min"),
            high_congestion_pct=("Congestion_Level", lambda x: (x == "High").mean() * 100),
            medium_congestion_pct=("Congestion_Level", lambda x: (x == "Medium").mean() * 100),
            low_congestion_pct=("Congestion_Level", lambda x: (x == "Low").mean() * 100),
            lanes=("Lanes", "first"),
            capacity=("Road_Capacity_veh_hr", "first"),
            road_type=("Road_Type", "first")
        ).round(2).reset_index().to_dict(orient="records")

        summary = {
            "total_records": total_records,
            "monitored_locations": len(loc_comp),
            "date_range": {
                "start": str(df["Timestamp"].min()),
                "end": str(df["Timestamp"].max())
            },
            "kpis": {
                "avg_traffic_volume": round(float(df["Traffic_Volume_veh_hr"].mean()), 1),
                "avg_speed_kmph": round(float(df["Average_Speed_kmph"].mean()), 1),
                "avg_tc_ratio": round(float(df["Traffic_to_Capacity_Ratio"].mean()), 3),
                "total_accidents": int(df["Accident_Reported"].sum()),
                "total_events": int(df["Local_Event"].sum())
            },
            "congestion_distribution": {
                "counts": cl_counts,
                "percentages": cl_pcts
            },
            "hourly_trends": hourly_grp,
            "day_of_week_trends": dow_grp,
            "weather_impact": weather_grp,
            "road_condition_impact": road_grp,
            "peak_hour_comparison": peak_comp,
            "accident_comparison": accident_comp,
            "local_event_comparison": event_comp,
            "locations_comparison": loc_comp
        }
        
        self._summary_cache = summary
        return summary

    def get_location_analytics(self, location_name: str) -> Dict[str, Any]:
        """Provides drill-down analytics for a specific monitored location."""
        df = self.df[self.df["Location"].str.lower() == location_name.lower()]
        if df.empty:
            # Try by Location_ID
            df = self.df[self.df["Location_ID"].str.lower() == location_name.lower()]
            if df.empty:
                return {"status": "error", "message": f"Location '{location_name}' not found"}

        actual_name = df["Location"].iloc[0]
        actual_id = df["Location_ID"].iloc[0]
        total_records = len(df)

        cl_counts = df["Congestion_Level"].value_counts().to_dict()
        cl_pcts = {k: round(v / total_records * 100, 2) for k, v in cl_counts.items()}

        hourly = df.groupby("Hour").agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            avg_tc_ratio=("Traffic_to_Capacity_Ratio", "mean"),
            high_congestion_pct=("Congestion_Level", lambda x: (x == "High").mean() * 100)
        ).round(2).reset_index().to_dict(orient="records")

        weather_stats = df.groupby("Weather").agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            records=("Congestion_Level", "count")
        ).round(2).reset_index().to_dict(orient="records")

        road_stats = df.groupby("Road_Condition").agg(
            avg_volume=("Traffic_Volume_veh_hr", "mean"),
            avg_speed=("Average_Speed_kmph", "mean"),
            records=("Congestion_Level", "count")
        ).round(2).reset_index().to_dict(orient="records")

        return {
            "status": "success",
            "location_id": actual_id,
            "location_name": actual_name,
            "records": total_records,
            "road_type": df["Road_Type"].iloc[0],
            "lanes": int(df["Lanes"].iloc[0]),
            "capacity": int(df["Road_Capacity_veh_hr"].iloc[0]),
            "kpis": {
                "avg_volume": round(float(df["Traffic_Volume_veh_hr"].mean()), 1),
                "max_volume": int(df["Traffic_Volume_veh_hr"].max()),
                "avg_speed": round(float(df["Average_Speed_kmph"].mean()), 1),
                "min_speed": round(float(df["Average_Speed_kmph"].min()), 1),
                "avg_tc_ratio": round(float(df["Traffic_to_Capacity_Ratio"].mean()), 3),
                "accident_count": int(df["Accident_Reported"].sum()),
                "event_count": int(df["Local_Event"].sum())
            },
            "congestion_distribution": {
                "counts": cl_counts,
                "percentages": cl_pcts
            },
            "hourly_profile": hourly,
            "weather_breakdown": weather_stats,
            "road_breakdown": road_stats
        }

    # ------------------------------------------------------------------
    # Dataset Overview / Analytics / Preview
    # Served from this service's already-loaded processed dataset (self.df) —
    # the SAME single data source the Historical Map and /api/analytics use.
    # ------------------------------------------------------------------

    def get_dataset_overview(self) -> Dict[str, Any]:
        """Schema overview derived from the shared processed dataset (no separate loader)."""
        try:
            from backend.services.dataset_service import analyze_schema
        except ImportError:
            from services.dataset_service import analyze_schema
        if self._overview_cache is None:
            ov = analyze_schema(self.df, "TRAFFIQ processed traffic dataset (data/traffic_processed.csv) — historical/static CSV data")
            ov["is_default_dataset"] = True
            ov["data_nature"] = "Historical / static dataset — not live or real-time traffic"
            ov["shared_source_note"] = "Served from the same in-memory dataset used by the Historical Map and all /api/analytics endpoints."
            self._overview_cache = ov
        return self._overview_cache

    def get_dataset_analytics(self) -> Dict[str, Any]:
        """Full automatic analytics derived from the shared processed dataset."""
        try:
            from backend.services.dataset_service import _resolve_columns, compute_full_analytics
        except ImportError:
            from services.dataset_service import _resolve_columns, compute_full_analytics
        if self._analytics_cache is None:
            self._analytics_cache = compute_full_analytics(self.df, _resolve_columns(self.df))
        return self._analytics_cache

    def get_dataset_preview(self, search: Optional[str], limit: int) -> Dict[str, Any]:
        """Preview/search rows derived from the shared processed dataset."""
        try:
            from backend.services.dataset_service import build_preview
        except ImportError:
            from services.dataset_service import build_preview
        return build_preview(self.df, search, limit)

analytics_service = AnalyticsService()
