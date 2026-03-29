from __future__ import annotations

from typing import Callable
import pandas as pd

from src.api import get_base_fpl_tables, get_all_player_histories_df

ProgressCallback = Callable[[float, str], None]


def _emit_progress(progress_callback: ProgressCallback | None, value: float, message: str) -> None:
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, value)), message)


def build_players_master_table(use_cache: bool = True) -> pd.DataFrame:
    tables = get_base_fpl_tables(use_cache=use_cache)
    players = tables["players"].copy()
    teams = tables["teams"].copy()

    if players.empty:
        return pd.DataFrame()

    keep_cols = [
        "id",
        "web_name",
        "full_name",
        "team",
        "position",
        "price_m",
        "status",
        "chance_of_playing_next_round",
    ]
    keep_cols = [col for col in keep_cols if col in players.columns]
    players = players[keep_cols].copy()

    if not teams.empty and "team" in players.columns and "team_id" in teams.columns:
        players = players.merge(
            teams[["team_id", "team_name", "team_short_name"]],
            left_on="team",
            right_on="team_id",
            how="left",
        )
        players = players.drop(columns=["team_id"], errors="ignore")
        players = players.rename(columns={"team_name": "team_name_current"})

    return players


def build_raw_history_table(
    use_cache: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    _emit_progress(progress_callback, 0.02, "Loading player master table...")
    players_master = build_players_master_table(use_cache=use_cache)

    if players_master.empty or "id" not in players_master.columns:
        return pd.DataFrame()

    player_ids = players_master["id"].dropna().astype(int).tolist()

    def history_progress(local_progress: float, message: str) -> None:
        mapped_progress = 0.08 + (0.72 * local_progress)
        _emit_progress(progress_callback, mapped_progress, message)

    history_df = get_all_player_histories_df(
        player_ids=player_ids,
        use_cache=use_cache,
        progress_callback=history_progress,
    )

    if history_df.empty:
        return pd.DataFrame()

    _emit_progress(progress_callback, 0.82, "Raw history table built.")
    return history_df


def build_clean_history_table(
    use_cache: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    _emit_progress(progress_callback, 0.01, "Starting preprocessing...")
    players_master = build_players_master_table(use_cache=use_cache)
    history = build_raw_history_table(use_cache=use_cache, progress_callback=progress_callback)

    if players_master.empty or history.empty:
        return pd.DataFrame()

    metadata_cols = [
        "id",
        "web_name",
        "full_name",
        "position",
        "price_m",
        "team_name_current",
        "team_short_name",
    ]
    metadata_cols = [col for col in metadata_cols if col in players_master.columns]

    _emit_progress(progress_callback, 0.86, "Merging player metadata...")
    history = history.merge(
        players_master[metadata_cols],
        left_on="player_id",
        right_on="id",
        how="left",
    )

    history = history.rename(
        columns={
            "web_name": "name",
            "price_m": "value",
            "team_name_current": "team",
        }
    )

    history = history.loc[:, ~history.columns.duplicated()].copy()

    if "round" not in history.columns:
        return pd.DataFrame()

    history = history[history["round"].notna()].copy()
    history["round"] = pd.to_numeric(history["round"], errors="coerce")
    history = history.dropna(subset=["round"])
    history["round"] = history["round"].astype(int)

    if "full_name" in history.columns and "name" in history.columns:
        history["name"] = history["full_name"].fillna(history["name"])

    if "was_home" not in history.columns:
        history["was_home"] = 0

    required_cols = [
        "name",
        "position",
        "team",
        "round",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "goals_scored",
        "minutes",
        "saves",
        "starts",
        "total_points",
        "value",
        "was_home",
    ]

    for col in required_cols:
        if col not in history.columns:
            history[col] = 0

    clean_df = history[required_cols].copy()
    clean_df = clean_df.loc[:, ~clean_df.columns.duplicated()].copy()

    numeric_cols = [
        "round",
        "assists",
        "clean_sheets",
        "goals_conceded",
        "goals_scored",
        "minutes",
        "saves",
        "starts",
        "total_points",
        "value",
        "was_home",
    ]

    for col in numeric_cols:
        if col in clean_df.columns:
            clean_df[col] = pd.to_numeric(clean_df[col], errors="coerce")

    clean_df = clean_df.dropna(subset=["name", "position", "round"]).copy()

    fill_zero_cols = [
        "assists",
        "clean_sheets",
        "goals_conceded",
        "goals_scored",
        "minutes",
        "saves",
        "starts",
        "total_points",
        "value",
        "was_home",
    ]

    existing_fill_cols = [col for col in fill_zero_cols if col in clean_df.columns]
    clean_df[existing_fill_cols] = clean_df[existing_fill_cols].fillna(0)

    clean_df["was_home"] = clean_df["was_home"].astype(int)
    clean_df = clean_df.sort_values(["name", "round"]).reset_index(drop=True)

    _emit_progress(progress_callback, 1.0, "Preprocessing complete.")
    return clean_df


def inspect_clean_history_table(use_cache: bool = True) -> pd.DataFrame:
    df = build_clean_history_table(use_cache=use_cache)
    print("Clean history shape:", df.shape)
    print(df.head())
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nDtypes:")
    print(df.dtypes)
    return df