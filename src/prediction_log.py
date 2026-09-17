from datetime import datetime
import numpy as np
import pandas as pd
from src.api import fetch_bootstrap_static, fetch_gameweek_live

def build_prediction_log_export(
    df: pd.DataFrame,
    target_gameweek: int | None = None,
) -> tuple[pd.DataFrame, int | None]:
    source_df = df.reset_index(drop=True).copy()

    def export_values(*column_names: str) -> pd.Series:
        for column_name in column_names:
            if column_name in source_df.columns:
                return source_df[column_name]
        return pd.Series("", index=source_df.index, dtype="object")

    gameweek = infer_prediction_target_gameweek(source_df)
    if gameweek is None or (target_gameweek is not None and target_gameweek != gameweek):
        raise ValueError("Current predictions must target the latest source round + 1.")

    position_values = export_values("position")
    model_values = position_values.astype("string").str.upper().map(
        {
            "GK": "Linear Regression",
            "DEF": "Linear Regression",
            "MID": "Linear Regression",
            "FWD": "LSTM",
        }
    )

    export_df = pd.DataFrame(index=source_df.index)
    export_df["Gameweek"] = gameweek if gameweek is not None else ""
    export_df["Export Timestamp"] = datetime.now().astimezone().isoformat(timespec="seconds")
    export_df["FPL Player ID"] = export_values("player_id")
    export_df["Player Name"] = export_values("name")
    export_df["Team"] = export_values("team")
    export_df["Position"] = position_values
    export_df["Price"] = pd.to_numeric(export_values("price_m"), errors="coerce").round(1)
    export_df["Predicted Points"] = pd.to_numeric(
        export_values("predicted_points"), errors="coerce"
    ).round(2)
    # Actual points are intentionally blank until the target gameweek finishes;
    # populating them for a future prediction would introduce data leakage.
    export_df["Actual Points"] = ""
    export_df["Error (Actual - Predicted)"] = ""
    export_df["Absolute Error"] = ""
    export_df["Squared Error"] = ""
    export_df["Risk Level"] = export_values("risk_level")
    export_df["Risk Flags"] = export_values("risk_flags")
    export_df["Next 3 FDR Avg"] = pd.to_numeric(
        export_values("next_3_fdr_avg"), errors="coerce"
    ).round(2)
    export_df["Next 5 FDR Avg"] = pd.to_numeric(
        export_values("next_5_fdr_avg"), errors="coerce"
    ).round(2)
    export_df["Model Used"] = model_values.fillna("")
    export_df["Latest Available Source Round"] = export_values(
        "source_round",
        "latest_available_source_round",
        "latest_available_round",
        "round",
    )
    export_df["Notes"] = ""

    return export_df, gameweek



def infer_prediction_target_gameweek(df: pd.DataFrame) -> int | None:
    for column in ("source_round", "latest_available_source_round", "latest_available_round", "round"):
        if column not in df:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        if values.empty or values.isna().any() or not np.isfinite(values).all() or (values % 1 != 0).any() or (values < 0).any():
            return None
        source = max(int(values.max()), int(df.attrs.get("latest_source_round", 0)))
        return source + 1 if 0 <= source < 38 else None
    return None


class ActualPointsUnavailable(ValueError):
    """The official event is unfinished or its points are unavailable."""


def validate_snapshot(df: pd.DataFrame) -> int:
    required = {"Gameweek", "Latest Available Source Round", "Predicted Points"}
    if df.empty or not required.issubset(df.columns):
        raise ValueError("Upload a Prediction Log CSV containing Gameweek, Latest Available Source Round and Predicted Points.")
    gw = pd.to_numeric(df["Gameweek"], errors="coerce")
    source = pd.to_numeric(df["Latest Available Source Round"], errors="coerce")
    predicted = pd.to_numeric(df["Predicted Points"], errors="coerce")
    if gw.isna().any() or gw.nunique() != 1 or not gw.between(1, 38).all() or (gw % 1 != 0).any():
        raise ValueError("The snapshot must contain exactly one valid target Gameweek.")
    if source.isna().any() or not np.isfinite(source).all() or (source % 1 != 0).any() or (source < 0).any() or (source >= gw).any():
        raise ValueError("Invalid pre-gameweek snapshot: every source round must be less than the target Gameweek. Predictions cannot be relabeled or regenerated.")
    if predicted.isna().any() or not np.isfinite(predicted).all():
        raise ValueError("Every row must contain a finite Predicted Points value.")
    return int(gw.iloc[0])


def complete_prediction_log(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    gameweek = validate_snapshot(df)
    bootstrap = fetch_bootstrap_static(use_cache=False)
    event = next((e for e in bootstrap.get("events", []) if e.get("id") == gameweek), None)
    if not event or event.get("finished") is not True or event.get("data_checked") is not True:
        raise ActualPointsUnavailable(f"Official final points for GW{gameweek} are not available yet. Please try again after the gameweek is completed and checked.")
    # Timestamps also guard against accidentally matching season-scoped IDs to a later season.
    if "Export Timestamp" not in df:
        raise ValueError("Export Timestamp is required to verify the snapshot season and pre-deadline timing.")
    timestamps = pd.to_datetime(df["Export Timestamp"], errors="coerce", utc=True)
    deadline = pd.to_datetime(event.get("deadline_time"), errors="coerce", utc=True)
    if timestamps.isna().any() or pd.isna(deadline):
        raise ValueError("Cannot verify the snapshot timestamp against the official deadline.")
    season_start = pd.Timestamp(year=deadline.year if deadline.month >= 7 else deadline.year - 1, month=7, day=1, tz="UTC")
    if (timestamps >= deadline).any() or (timestamps < season_start).any():
        raise ValueError("The snapshot must have been exported before the target deadline in the current FPL season.")
    players = bootstrap.get("elements", [])
    known_ids = {p["id"] for p in players}
    teams = {t["id"]: t["name"] for t in bootstrap.get("teams", [])}
    from src.constants import POSITION_MAP
    candidates = {}
    for player in players:
        names = {player.get("web_name", ""), f"{player.get('first_name', '')} {player.get('second_name', '')}".strip()}
        for name in names:
            key = (name.strip().casefold(), str(teams.get(player.get("team"), "")).strip().casefold(), POSITION_MAP.get(player.get("element_type")))
            candidates.setdefault(key, set()).add(player["id"])
    ids = []
    for _, row in df.iterrows():
        raw_id = str(row.get("FPL Player ID", "")).strip()
        if raw_id and raw_id.lower() != "nan":
            number = pd.to_numeric(raw_id, errors="coerce")
            pid = int(number) if pd.notna(number) and np.isfinite(number) and number % 1 == 0 else None
            ids.append(pid if pid in known_ids else None)
        else:
            key = (str(row.get("Player Name", "")).strip().casefold(), str(row.get("Team", "")).strip().casefold(), str(row.get("Position", "")).strip().upper())
            matches = candidates.get(key, set())
            ids.append(next(iter(matches)) if len(matches) == 1 else None)
    live = fetch_gameweek_live(gameweek)
    totals = {}
    for player in live["elements"]:
        value = pd.to_numeric(player.get("stats", {}).get("total_points"), errors="coerce")
        if pd.notna(value) and np.isfinite(value):
            totals[player["id"]] = value
    if not totals:
        raise ActualPointsUnavailable(f"Official points for GW{gameweek} are not available yet. Please try again later.")
    result = df.copy(deep=True)
    actual = pd.Series([totals.get(pid, np.nan) for pid in ids], index=df.index)
    error = actual - pd.to_numeric(df["Predicted Points"])
    result["Actual Points"] = actual
    result["Error (Actual - Predicted)"] = error
    result["Absolute Error"] = error.abs()
    result["Squared Error"] = error ** 2
    return result, gameweek, int(actual.isna().sum())
