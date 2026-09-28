/**
 * TRAFFIQ API Client Service
 */

const DEFAULT_RENDER_API = 'https://traffic-prediction-1-k5oc.onrender.com/api';

const getApiBase = (): string => {
  // 1. Explicit environment variable takes highest priority
  const envBase = (import.meta as any).env?.VITE_API_BASE;
  if (envBase && typeof envBase === 'string' && envBase.trim().length > 0) {
    return envBase.replace(/\/+$/, '');
  }

  // 2. In browser environments
  if (typeof window !== 'undefined') {
    const host = window.location.hostname;
    // Local dev: use relative /api proxy
    if (host === 'localhost' || host === '127.0.0.1' || host === '0.0.0.0') {
      return '/api';
    }
    // Remote cloud deployment (e.g. Vercel, Netlify): direct to Render backend
    return DEFAULT_RENDER_API;
  }

  return '/api';
};

const API_BASE = getApiBase();

export interface LocationItem {
  Location_ID: string;
  Location_Name: string;
  Latitude: number;
  Longitude: number;
  Geocoding_Source: string;
  Geocoding_Query: string;
  Geocoding_Status: string;
  Road_Type: string;
  Lanes: number;
  Road_Capacity: number;
  City?: string;
  State?: string;
}

export interface AnalyticsSummary {
  total_records: number;
  monitored_locations: number;
  date_range: { start: string; end: string };
  kpis: {
    avg_traffic_volume: number;
    avg_speed_kmph: number;
    avg_tc_ratio: number;
    total_accidents: number;
    total_events: number;
  };
  congestion_distribution: {
    counts: Record<string, number>;
    percentages: Record<string, number>;
  };
  hourly_trends: Array<{
    Hour: number;
    avg_volume: number;
    avg_speed: number;
    avg_tc_ratio: number;
    high_congestion_pct: number;
  }>;
  day_of_week_trends: Array<{
    DayOfWeek: number;
    Day_Name: string;
    avg_volume: number;
    avg_speed: number;
    high_congestion_pct: number;
  }>;
  weather_impact: Array<{
    Weather: string;
    avg_volume: number;
    avg_speed: number;
    avg_tc_ratio: number;
    high_congestion_pct: number;
    records: number;
  }>;
  road_condition_impact: Array<{
    Road_Condition: string;
    avg_volume: number;
    avg_speed: number;
    avg_tc_ratio: number;
    high_congestion_pct: number;
    records: number;
  }>;
  locations_comparison: Array<{
    Location: string;
    records: number;
    avg_volume: number;
    avg_speed: number;
    avg_tc_ratio: number;
    max_volume: number;
    min_speed: number;
    high_congestion_pct: number;
    medium_congestion_pct: number;
    low_congestion_pct: number;
    lanes: number;
    capacity: number;
    road_type: string;
  }>;
}

export interface PredictionResult {
  status: string;
  prediction: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
  base_ml_prediction: 'Low' | 'Medium' | 'High';
  is_critical_escalated: boolean;
  escalation_reason?: string;
  confidence: number;
  probabilities: Record<string, number>;
  estimated_metrics: {
    estimated_volume_veh_hr: number;
    estimated_tc_ratio: number;
    estimated_speed_kmph: number;
    road_capacity: number;
    lanes: number;
  };
  explainability: {
    summary: string;
    top_factors: Array<{
      factor: string;
      impact: string;
      importance_pct: number;
      description: string;
    }>;
  };
  recommendations?: {
    user_travel_advice: Array<{ type: string; badge: string; title: string; detail: string }>;
    traffic_management_recommendations: Array<{ category: string; priority: string; action: string; detail: string }>;
  };
}

export interface RouteIntelligenceResult {
  status: string;
  intelligence_type: string;
  data_honesty_notice: string;
  route: {
    origin: { id: string; name: string; lat: number; lon: number; road_type: string; capacity: number };
    destination: { id: string; name: string; lat: number; lon: number; road_type: string; capacity: number };
    distance_km: number;
    estimated_duration_min: number;
    geometry: {
      type: string;
      coordinates: Array<[number, number]>;
    };
    source: string;
  };
  origin_intelligence: {
    location_name: string;
    historical_kpis: any;
    historical_congestion_distribution: Record<string, number>;
    departure_ml_prediction: string;
    base_ml_prediction: string;
    is_critical_escalated: boolean;
    confidence: number;
    estimated_traffic_volume: number;
    estimated_speed_kmph: number;
    tc_ratio: number;
  };
  destination_intelligence: {
    location_name: string;
    historical_kpis: any;
    historical_congestion_distribution: Record<string, number>;
    arrival_ml_prediction: string;
    estimated_traffic_volume: number;
    estimated_speed_kmph: number;
    tc_ratio: number;
  };
  explainability: {
    summary: string;
    top_factors: Array<{
      factor: string;
      impact: string;
      importance_pct: number;
      description: string;
    }>;
  };
  recommendations: {
    user_travel_advice: Array<{ type: string; badge: string; title: string; detail: string }>;
    traffic_management_recommendations: Array<{ category: string; priority: string; action: string; detail: string }>;
  };
}

// ---------- Dataset & Analytics types ----------

export interface DatasetColumn {
  name: string;
  dtype: 'number' | 'text';
  missing: number;
  unique: number;
  sample: string | null;
}

export interface DatasetOverview {
  source: string;
  data_nature?: string;
  is_default_dataset?: boolean;
  row_count: number;
  column_count: number;
  missing_cells: number;
  duplicate_rows: number;
  unique_locations: number;
  location_column: string | null;
  timestamp_range: { available: boolean; column?: string; start?: string; end?: string; invalid_or_missing?: number };
  columns: DatasetColumn[];
  categorical_values: Array<{ column: string; unique: number; top_values: Array<{ value: string; count: number }> }>;
  numeric_summary: Array<{ column: string; mean: number | null; std: number | null; min: number | null; median: number | null; max: number | null }>;
}

export interface UnavailableSection {
  available: false;
  reason: string;
}

export interface FullAnalytics {
  status: string;
  data_nature: string;
  analytics: Record<string, any>;
}

export interface DatasetPreview {
  columns: string[];
  rows: any[][];
  returned: number;
  matched_total: number;
  dataset_total: number;
  truncated: boolean;
}

export interface ModelPerformance {
  status: string;
  data_basis_notice: string;
  data_nature: string;
  best_model: string;
  models: Record<string, { accuracy: number | null; weighted_f1: number | null; macro_f1: number | null; selected: boolean }>;
  classes: string[];
  overall: {
    accuracy: number | null;
    precision_macro: number | null;
    precision_weighted: number | null;
    recall_macro: number | null;
    recall_weighted: number | null;
    f1_macro: number | null;
    f1_weighted: number | null;
  };
  confusion_matrix: { labels: string[]; matrix: number[][] };
  per_class: Record<string, { precision: number; recall: number; f1_score: number; support: number }>;
  feature_importance: Array<{ feature: string; percentage: number }>;
  leakage_prevention: { excluded_features: string[]; explanation: string };
  test_split: { size: number; fraction: number; strategy: string };
}

export interface UploadAnalysis {
  status: string;
  dataset_id: string;
  filename: string;
  data_nature: string;
  overview: DatasetOverview;
  analytics: Record<string, any>;
  ml_feature_availability: {
    target_available: boolean;
    features: Array<{ feature: string; status: 'present' | 'derivable' | 'missing'; note?: string }>;
    missing_required: string[];
    notice: string;
  };
  geography: { available: boolean; method: string | null; locations: any[]; notice: string | null };
  sampled_note: string | null;
}

// ---------- Global place search (any real-world location) ----------

export interface GeocodeResult {
  status: 'ok' | 'not_found' | 'error';
  query: string;
  lat: number | null;
  lon: number | null;
  display_name: string | null;
  in_dataset: boolean;
  dataset_location?: LocationItem | null;
}

export interface SimulationResult {
  status: string;
  simulation_mode: string;
  disclaimer: string;
  inputs: any;
  comparison: {
    baseline: {
      congestion_level: string;
      estimated_volume: number;
      tc_ratio: number;
      estimated_speed_kmph: number;
    };
    simulated: {
      congestion_level: string;
      estimated_volume: number;
      tc_ratio: number;
      estimated_speed_kmph: number;
      is_critical_escalated: boolean;
      escalation_reason?: string;
    };
    deltas: {
      volume_change_veh_hr: number;
      speed_loss_kmph: number;
      tc_ratio_increase: number;
    };
  };
  explainability: any;
  recommendations: any;
}

export const api = {
  getLocations: async (): Promise<LocationItem[]> => {
    const res = await fetch(`${API_BASE}/locations`);
    if (!res.ok) throw new Error('Failed to fetch locations');
    return res.json();
  },

  getSummary: async (): Promise<AnalyticsSummary> => {
    const res = await fetch(`${API_BASE}/analytics/summary`);
    if (!res.ok) throw new Error('Failed to fetch analytics summary');
    return res.json();
  },

  getLocationAnalytics: async (location: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/analytics/location/${encodeURIComponent(location)}`);
    if (!res.ok) throw new Error(`Failed to fetch analytics for ${location}`);
    return res.json();
  },

  predict: async (payload: any): Promise<PredictionResult> => {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Prediction API error');
    return res.json();
  },

  getRoute: async (origin: string, destination: string): Promise<any> => {
    const res = await fetch(`${API_BASE}/route`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ origin, destination })
    });
    if (!res.ok) throw new Error('Route API error');
    return res.json();
  },

  getRouteIntelligence: async (payload: any): Promise<RouteIntelligenceResult> => {
    const res = await fetch(`${API_BASE}/route-intelligence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Route Intelligence API error');
    return res.json();
  },

  runSimulation: async (payload: any): Promise<SimulationResult> => {
    const res = await fetch(`${API_BASE}/simulation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Simulation API error');
    return res.json();
  },

  // ---------- Dataset & Analytics (automatic dataset analysis) ----------

  getDatasetOverview: async (): Promise<DatasetOverview> => {
    const res = await fetch(`${API_BASE}/dataset/overview`);
    if (!res.ok) throw new Error('Failed to fetch dataset overview');
    return res.json();
  },

  getFullAnalytics: async (): Promise<FullAnalytics> => {
    const res = await fetch(`${API_BASE}/dataset/analytics/full`);
    if (!res.ok) throw new Error('Failed to fetch full analytics');
    return res.json();
  },

  getDatasetPreview: async (search: string, limit: number): Promise<DatasetPreview> => {
    const res = await fetch(`${API_BASE}/dataset/preview?search=${encodeURIComponent(search)}&limit=${limit}`);
    if (!res.ok) throw new Error('Failed to fetch dataset preview');
    return res.json();
  },

  getModelPerformance: async (): Promise<ModelPerformance> => {
    const res = await fetch(`${API_BASE}/model/performance`);
    if (!res.ok) throw new Error('Failed to fetch model performance');
    return res.json();
  },

  // ---------- Global place search & coordinate routing (real-world locations) ----------

  geocodeSearch: async (q: string): Promise<GeocodeResult> => {
    const res = await fetch(`${API_BASE}/geocode?q=${encodeURIComponent(q)}`);
    if (!res.ok) throw new Error('Geocoding service unavailable');
    return res.json();
  },

  getRouteByPoints: async (origin: { lat: number; lon: number; name?: string }, dest: { lat: number; lon: number; name?: string }): Promise<any> => {
    const res = await fetch(`${API_BASE}/route/points`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        origin_lat: origin.lat,
        origin_lon: origin.lon,
        dest_lat: dest.lat,
        dest_lon: dest.lon,
        origin_name: origin.name,
        destination_name: dest.name
      })
    });
    if (!res.ok) throw new Error('Route API error');
    return res.json();
  },

  uploadDataset: async (file: File): Promise<UploadAnalysis> => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${API_BASE}/dataset/upload`, { method: 'POST', body: form });
    if (!res.ok) {
      let detail = 'Upload analysis failed';
      try { detail = (await res.json()).detail || detail; } catch { /* keep default */ }
      throw new Error(detail);
    }
    return res.json();
  }
};
