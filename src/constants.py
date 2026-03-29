from __future__ import annotations

from pathlib import Path

# -----------------------------
# Project paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_CACHE_DIR = PROJECT_ROOT / "data_cache"

# Ensure cache directory exists
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Official FPL API endpoints
# -----------------------------
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
BOOTSTRAP_STATIC_URL = f"{FPL_BASE_URL}/bootstrap-static/"
FIXTURES_URL = f"{FPL_BASE_URL}/fixtures/"
ELEMENT_SUMMARY_URL = f"{FPL_BASE_URL}/element-summary/{{player_id}}/"

# -----------------------------
# Timeouts / caching
# -----------------------------
REQUEST_TIMEOUT_SECONDS = 30

# Optional disk cache files
BOOTSTRAP_CACHE_FILE = DATA_CACHE_DIR / "bootstrap_static.json"
FIXTURES_CACHE_FILE = DATA_CACHE_DIR / "fixtures.json"
ELEMENT_SUMMARY_CACHE_DIR = DATA_CACHE_DIR / "element_summaries"
ELEMENT_SUMMARY_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# FPL position mapping
# element_type in bootstrap-static:
# 1 = GK, 2 = DEF, 3 = MID, 4 = FWD
# -----------------------------
POSITION_MAP = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}

# -----------------------------
# Final model feature lists
# IMPORTANT:
# Keep the exact order used in training.
# -----------------------------
GK_FEATURES = [
    "minutes",
    "saves",
    "goals_conceded",
    "clean_sheets",
    "value",
    "was_home",
    "total_points_lag1",
    "minutes_lag1",
    "saves_lag1",
    "goals_conceded_lag1",
    "total_points_rolling3",
    "minutes_rolling3",
    "points_std3",
    "played_last_gw",
]

DEF_FEATURES = [
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "value",
    "was_home",
    "total_points_lag1",
    "goals_scored_lag1",
    "assists_lag1",
    "clean_sheets_lag1",
    "total_points_rolling3",
    "goals_scored_rolling3",
    "assists_rolling3",
    "points_std3",
    "played_last_gw",
]

MID_FEATURES = [
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "value",
    "was_home",
    "total_points_lag1",
    "goals_scored_lag1",
    "assists_lag1",
    "clean_sheets_lag1",
    "total_points_rolling3",
    "goals_scored_rolling3",
    "assists_rolling3",
    "points_std3",
    "played_last_gw",
]

FWD_FEATURES = [
    "minutes",
    "goals_scored",
    "assists",
    "value",
    "was_home",
    "total_points_lag1",
    "goals_scored_lag1",
    "assists_lag1",
    "total_points_rolling3",
    "goals_scored_rolling3",
    "assists_rolling3",
    "points_std3",
    "played_last_gw",
]

FEATURES_BY_POSITION = {
    "GK": GK_FEATURES,
    "DEF": DEF_FEATURES,
    "MID": MID_FEATURES,
    "FWD": FWD_FEATURES,
}

# -----------------------------
# Model filenames
# Adjust these later if your filenames differ.
# -----------------------------
GK_MODEL_PATH = MODELS_DIR / "gk_linear.pkl"
DEF_MODEL_PATH = MODELS_DIR / "def_linear.pkl"
MID_MODEL_PATH = MODELS_DIR / "mid_linear.pkl"
FWD_MODEL_PATH = MODELS_DIR / "fwd_lstm.keras"

MODEL_PATHS = {
    "GK": GK_MODEL_PATH,
    "DEF": DEF_MODEL_PATH,
    "MID": MID_MODEL_PATH,
    "FWD": FWD_MODEL_PATH,
}