from __future__ import annotations

from datetime import datetime
from typing import Callable
import re
import unicodedata

import pandas as pd
import pulp

from src.preprocess import build_players_master_table

ProgressCallback = Callable[[float, str], None]

VAASTAV_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    value: float,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, value)), message)


def _normalize_name(value: str) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = value.encode("ascii", "ignore").decode("ascii")
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s]", "", value)
    value = re.sub(r"\s+", " ", value)
    return value


def infer_previous_season_folder(today: datetime | None = None) -> str:
    """
    Infer the previous completed FPL season folder, e.g. '2024-25'.
    """
    if today is None:
        today = datetime.utcnow()

    year = today.year
    month = today.month

    if month >= 7:
        # current season starts this calendar year, previous season starts year-1
        start_year = year - 1
    else:
        # current season started last calendar year, previous season starts year-2
        start_year = year - 2

    end_short = str((start_year + 1) % 100).zfill(2)
    return f"{start_year}-{end_short}"


def load_previous_season_players_raw(
    season_folder: str | None = None,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    """
    Load archived previous-season players_raw.csv from vaastav repo.
    """
    if season_folder is None:
        season_folder = infer_previous_season_folder()

    _emit_progress(progress_callback, 0.05, f"Loading archived previous-season data: {season_folder}...")
    url = f"{VAASTAV_RAW_BASE}/{season_folder}/players_raw.csv"
    prev_df = pd.read_csv(url)

    if prev_df.empty:
        return pd.DataFrame()

    return prev_df


def build_previous_season_summary(
    season_folder: str | None = None,
    progress_callback: ProgressCallback | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build a summary table from archived previous-season players_raw.csv.
    """
    prev_df = load_previous_season_players_raw(
        season_folder=season_folder,
        progress_callback=progress_callback,
    )

    if prev_df.empty:
        return pd.DataFrame()

    _log(f"Archived previous-season rows loaded: {len(prev_df)}", verbose)
    _emit_progress(progress_callback, 0.18, "Building previous-season summary table...")

    # Name construction
    first_name_col = "first_name" if "first_name" in prev_df.columns else None
    second_name_col = "second_name" if "second_name" in prev_df.columns else None
    web_name_col = "web_name" if "web_name" in prev_df.columns else None

    if first_name_col and second_name_col:
        prev_df["full_name"] = (
            prev_df[first_name_col].fillna("").astype(str).str.strip()
            + " "
            + prev_df[second_name_col].fillna("").astype(str).str.strip()
        ).str.strip()
    else:
        prev_df["full_name"] = prev_df.get(web_name_col, "").astype(str)

    prev_df["web_name_safe"] = prev_df.get(web_name_col, prev_df["full_name"]).astype(str)

    # Numeric fields
    numeric_defaults = {
        "total_points": 0.0,
        "points_per_game": 0.0,
        "minutes": 0.0,
        "goals_scored": 0.0,
        "assists": 0.0,
        "clean_sheets": 0.0,
        "goals_conceded": 0.0,
        "saves": 0.0,
        "starts": 0.0,
        "now_cost": 0.0,
    }

    for col, default in numeric_defaults.items():
        if col not in prev_df.columns:
            prev_df[col] = default
        prev_df[col] = pd.to_numeric(prev_df[col], errors="coerce").fillna(default)

    # Approximate appearances
    # Prefer starts if present, otherwise derive from minutes.
    appearances = prev_df["starts"].copy()
    appearances = appearances.where(appearances > 0, (prev_df["minutes"] / 70.0).round())
    appearances = appearances.where(appearances > 0, 1)
    prev_df["appearances"] = appearances

    prev_df["total_points_sum"] = prev_df["total_points"]
    prev_df["total_points_avg"] = prev_df["points_per_game"]
    prev_df["total_points_std"] = 0.0  # no direct season std in players_raw; neutral proxy
    prev_df["minutes_sum"] = prev_df["minutes"]
    prev_df["minutes_avg"] = prev_df["minutes_sum"] / prev_df["appearances"]
    prev_df["goals_scored_sum"] = prev_df["goals_scored"]
    prev_df["assists_sum"] = prev_df["assists"]
    prev_df["clean_sheets_sum"] = prev_df["clean_sheets"]
    prev_df["goals_conceded_sum"] = prev_df["goals_conceded"]
    prev_df["saves_sum"] = prev_df["saves"]
    prev_df["starts_sum"] = prev_df["starts"]
    prev_df["final_value_prev"] = prev_df["now_cost"]
    prev_df["points_per_minute"] = prev_df["total_points_sum"] / prev_df["minutes_sum"].replace(0, pd.NA)
    prev_df["points_per_minute"] = prev_df["points_per_minute"].fillna(0.0)
    prev_df["start_rate"] = prev_df["starts_sum"] / prev_df["appearances"].replace(0, pd.NA)
    prev_df["start_rate"] = prev_df["start_rate"].fillna(0.0)
    prev_df["minutes_per_appearance"] = prev_df["minutes_sum"] / prev_df["appearances"].replace(0, pd.NA)
    prev_df["minutes_per_appearance"] = prev_df["minutes_per_appearance"].fillna(0.0)
    prev_df["value_efficiency"] = prev_df["total_points_sum"] / prev_df["final_value_prev"].replace(0, pd.NA)
    prev_df["value_efficiency"] = prev_df["value_efficiency"].fillna(0.0)

    prev_df["match_key_full"] = prev_df["full_name"].map(_normalize_name)
    prev_df["match_key_web"] = prev_df["web_name_safe"].map(_normalize_name)

    keep_cols = [
        "full_name",
        "web_name_safe",
        "match_key_full",
        "match_key_web",
        "appearances",
        "total_points_sum",
        "total_points_avg",
        "total_points_std",
        "minutes_sum",
        "minutes_avg",
        "goals_scored_sum",
        "assists_sum",
        "clean_sheets_sum",
        "goals_conceded_sum",
        "saves_sum",
        "starts_sum",
        "final_value_prev",
        "points_per_minute",
        "start_rate",
        "minutes_per_appearance",
        "value_efficiency",
    ]

    summary_df = prev_df[keep_cols].copy()

    # Prefer best historical row if duplicate names somehow appear
    summary_df = summary_df.sort_values(
        ["match_key_full", "total_points_sum"],
        ascending=[True, False]
    ).drop_duplicates(subset=["match_key_full"], keep="first")

    summary_web_df = summary_df.sort_values(
        ["match_key_web", "total_points_sum"],
        ascending=[True, False]
    ).drop_duplicates(subset=["match_key_web"], keep="first")

    _emit_progress(progress_callback, 0.28, "Previous-season summary ready.")
    return summary_df, summary_web_df


def build_gw1_candidate_pool(
    use_cache: bool = True,
    include_unmatched: bool = True,
    unmatched_penalty: float = 0.85,
    season_folder: str | None = None,
    progress_callback: ProgressCallback | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build GW1 candidate pool by merging live current pool with archived previous-season summary.
    """
    _emit_progress(progress_callback, 0.01, "Loading current live player pool...")
    players_df = build_players_master_table(use_cache=use_cache).copy()

    if players_df.empty:
        return pd.DataFrame()

    keep_cols = [
        "id",
        "web_name",
        "full_name",
        "position",
        "team_name_current",
        "price_m",
        "status",
        "chance_of_playing_next_round",
    ]
    keep_cols = [col for col in keep_cols if col in players_df.columns]
    players_df = players_df[keep_cols].copy()

    players_df = players_df.dropna(subset=["position", "price_m"]).copy()
    players_df = players_df[players_df["position"].isin(["GK", "DEF", "MID", "FWD"])].copy()

    players_df["match_key_full"] = players_df["full_name"].fillna("").astype(str).map(_normalize_name)
    players_df["match_key_web"] = players_df["web_name"].fillna("").astype(str).map(_normalize_name)

    _log(f"Current player pool size: {len(players_df)}", verbose)
    _emit_progress(progress_callback, 0.10, "Loading archived previous-season summary...")

    summary_full_df, summary_web_df = build_previous_season_summary(
        season_folder=season_folder,
        progress_callback=progress_callback,
        verbose=verbose,
    )

    # Match by full name first
    merged_full = players_df.merge(
        summary_full_df,
        on="match_key_full",
        how="left",
        suffixes=("", "_prev"),
    )

    # For unmatched rows, try web_name fallback
    unmatched_mask = merged_full["total_points_sum"].isna()
    unmatched_current = merged_full.loc[unmatched_mask, players_df.columns].copy()

    if not unmatched_current.empty:
        merged_web = unmatched_current.merge(
            summary_web_df,
            on="match_key_web",
            how="left",
            suffixes=("", "_prev"),
        )

        matched_cols = [
            "appearances",
            "total_points_sum",
            "total_points_avg",
            "total_points_std",
            "minutes_sum",
            "minutes_avg",
            "goals_scored_sum",
            "assists_sum",
            "clean_sheets_sum",
            "goals_conceded_sum",
            "saves_sum",
            "starts_sum",
            "final_value_prev",
            "points_per_minute",
            "start_rate",
            "minutes_per_appearance",
            "value_efficiency",
        ]

        for col in matched_cols:
            merged_full.loc[unmatched_mask, col] = merged_web[col].values

    candidate_pool = merged_full.copy()

    candidate_pool = candidate_pool.rename(
        columns={
            "team_name_current": "team",
            "web_name": "name",
        }
    )

    candidate_pool["matched_previous_season"] = candidate_pool["total_points_sum"].notna().astype(int)

    if not include_unmatched:
        candidate_pool = candidate_pool[candidate_pool["matched_previous_season"] == 1].copy()

    _emit_progress(progress_callback, 0.45, "Applying fallback values for unmatched players...")

    matched_df = candidate_pool[candidate_pool["matched_previous_season"] == 1].copy()

    fallback_cols = [
        "appearances",
        "total_points_sum",
        "total_points_avg",
        "total_points_std",
        "minutes_sum",
        "minutes_avg",
        "goals_scored_sum",
        "assists_sum",
        "clean_sheets_sum",
        "goals_conceded_sum",
        "saves_sum",
        "starts_sum",
        "final_value_prev",
        "points_per_minute",
        "start_rate",
        "minutes_per_appearance",
        "value_efficiency",
    ]

    if not matched_df.empty:
        position_medians = (
            matched_df.groupby("position")[fallback_cols]
            .median(numeric_only=True)
            .reset_index()
        )
    else:
        position_medians = pd.DataFrame({
            "position": ["GK", "DEF", "MID", "FWD"],
            "appearances": [20, 20, 20, 20],
            "total_points_sum": [80, 90, 110, 100],
            "total_points_avg": [3.5, 3.8, 4.2, 4.0],
            "total_points_std": [0.0, 0.0, 0.0, 0.0],
            "minutes_sum": [1800, 1800, 1800, 1600],
            "minutes_avg": [70, 70, 70, 65],
            "goals_scored_sum": [0, 2, 5, 8],
            "assists_sum": [0, 3, 5, 3],
            "clean_sheets_sum": [8, 8, 4, 0],
            "goals_conceded_sum": [40, 40, 0, 0],
            "saves_sum": [80, 0, 0, 0],
            "starts_sum": [20, 20, 20, 18],
            "final_value_prev": [50, 50, 70, 70],
            "points_per_minute": [0.04, 0.04, 0.05, 0.05],
            "start_rate": [0.85, 0.85, 0.85, 0.80],
            "minutes_per_appearance": [70, 70, 70, 65],
            "value_efficiency": [1.8, 1.8, 2.0, 1.9],
        })

    candidate_pool = candidate_pool.merge(
        position_medians,
        on="position",
        how="left",
        suffixes=("", "_position_median"),
    )

    for col in fallback_cols:
        median_col = f"{col}_position_median"
        candidate_pool[col] = pd.to_numeric(candidate_pool[col], errors="coerce")
        candidate_pool[median_col] = pd.to_numeric(candidate_pool[median_col], errors="coerce")
        candidate_pool[col] = candidate_pool[col].fillna(candidate_pool[median_col])

    chance_series = pd.to_numeric(candidate_pool["chance_of_playing_next_round"], errors="coerce")
    chance_factor = chance_series.fillna(100) / 100.0
    status_penalty = candidate_pool["status"].astype(str).str.lower().map({
        "a": 1.00,
        "d": 0.85,
        "i": 0.60,
        "u": 0.50,
        "s": 0.75,
    }).fillna(0.90)

    candidate_pool["availability_factor"] = chance_factor * status_penalty
    candidate_pool["matched_penalty_factor"] = candidate_pool["matched_previous_season"].map({
        1: 1.00,
        0: unmatched_penalty,
    })

    score_features = [
        "total_points_sum",
        "total_points_avg",
        "total_points_std",
        "minutes_per_appearance",
        "start_rate",
        "value_efficiency",
    ]

    _emit_progress(progress_callback, 0.68, "Calculating hybrid scores...")

    for col in score_features:
        candidate_pool[col] = pd.to_numeric(candidate_pool[col], errors="coerce").fillna(0.0)
        min_val = candidate_pool[col].min()
        max_val = candidate_pool[col].max()
        if max_val != min_val:
            candidate_pool[f"{col}_norm"] = (candidate_pool[col] - min_val) / (max_val - min_val)
        else:
            candidate_pool[f"{col}_norm"] = 0.0

    candidate_pool["consistency_score"] = 1 - candidate_pool["total_points_std_norm"]

    candidate_pool["hybrid_score_raw"] = (
        0.35 * candidate_pool["total_points_avg_norm"] +
        0.20 * candidate_pool["value_efficiency_norm"] +
        0.15 * candidate_pool["start_rate_norm"] +
        0.15 * candidate_pool["minutes_per_appearance_norm"] +
        0.10 * candidate_pool["total_points_sum_norm"] +
        0.05 * candidate_pool["consistency_score"]
    )

    candidate_pool["hybrid_score"] = (
        candidate_pool["hybrid_score_raw"]
        * candidate_pool["availability_factor"]
        * candidate_pool["matched_penalty_factor"]
    )

    candidate_pool["value"] = (pd.to_numeric(candidate_pool["price_m"], errors="coerce").fillna(0.0) * 10).round(0)

    drop_helper_cols = [col for col in candidate_pool.columns if col.endswith("_position_median")]
    candidate_pool = candidate_pool.drop(columns=drop_helper_cols, errors="ignore")

    candidate_pool = candidate_pool.sort_values(
        ["matched_previous_season", "hybrid_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    _emit_progress(progress_callback, 0.80, "GW1 candidate pool ready.")
    return candidate_pool


def optimize_gw1_hybrid_squad(
    candidate_pool: pd.DataFrame,
    budget_limit: int = 1000,
    club_limit: int = 3,
    verbose: bool = True,
) -> tuple[pd.DataFrame, float, float, str]:
    if candidate_pool.empty:
        return pd.DataFrame(), 0.0, 0.0, "Empty"

    pool = candidate_pool.reset_index(drop=True).copy()
    problem = pulp.LpProblem("GW1_Hybrid_Squad", pulp.LpMaximize)

    player_vars = {
        i: pulp.LpVariable(f"gw1_player_{i}", cat="Binary")
        for i in pool.index
    }

    problem += pulp.lpSum(pool.loc[i, "hybrid_score"] * player_vars[i] for i in pool.index)

    problem += pulp.lpSum(player_vars[i] for i in pool.index) == 15
    problem += pulp.lpSum(player_vars[i] for i in pool.index if pool.loc[i, "position"] == "GK") == 2
    problem += pulp.lpSum(player_vars[i] for i in pool.index if pool.loc[i, "position"] == "DEF") == 5
    problem += pulp.lpSum(player_vars[i] for i in pool.index if pool.loc[i, "position"] == "MID") == 5
    problem += pulp.lpSum(player_vars[i] for i in pool.index if pool.loc[i, "position"] == "FWD") == 3

    problem += pulp.lpSum(pool.loc[i, "value"] * player_vars[i] for i in pool.index) <= budget_limit

    for club in pool["team"].dropna().unique().tolist():
        problem += pulp.lpSum(
            player_vars[i]
            for i in pool.index
            if pool.loc[i, "team"] == club
        ) <= club_limit

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[problem.status]
    selected_indices = [i for i in pool.index if player_vars[i].value() == 1]

    selected_squad = pool.loc[selected_indices].copy().reset_index(drop=True)
    total_cost = float(selected_squad["value"].sum())
    total_score = float(selected_squad["hybrid_score"].sum())

    return selected_squad, total_cost, total_score, status


def build_gw1_hybrid_outputs(
    use_cache: bool = True,
    include_unmatched: bool = True,
    unmatched_penalty: float = 0.85,
    season_folder: str | None = None,
    progress_callback: ProgressCallback | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    candidate_pool = build_gw1_candidate_pool(
        use_cache=use_cache,
        include_unmatched=include_unmatched,
        unmatched_penalty=unmatched_penalty,
        season_folder=season_folder,
        progress_callback=progress_callback,
        verbose=verbose,
    )

    if candidate_pool.empty:
        return {
            "candidate_pool": pd.DataFrame(),
            "hybrid_squad": pd.DataFrame(),
            "summary": pd.DataFrame(),
        }

    _emit_progress(progress_callback, 0.92, "Optimizing hybrid GW1 squad...")
    hybrid_squad, hybrid_cost, hybrid_score, hybrid_status = optimize_gw1_hybrid_squad(
        candidate_pool=candidate_pool,
        budget_limit=1000,
        club_limit=3,
        verbose=verbose,
    )

    summary = pd.DataFrame({
        "Method": ["Hybrid Score"],
        "Solve_Status": [hybrid_status],
        "Total_Cost": [hybrid_cost],
        "Total_Score": [hybrid_score],
        "Matched_Players_In_Pool": [int(candidate_pool["matched_previous_season"].sum())],
        "Unmatched_Players_In_Pool": [int((candidate_pool["matched_previous_season"] == 0).sum())],
        "Previous_Season_Source": [season_folder or infer_previous_season_folder()],
    })

    _emit_progress(progress_callback, 1.0, "GW1 hybrid squad ready.")
    return {
        "candidate_pool": candidate_pool,
        "hybrid_squad": hybrid_squad,
        "summary": summary,
    }


def inspect_gw1_builder(
    use_cache: bool = True,
    include_unmatched: bool = True,
    unmatched_penalty: float = 0.85,
    season_folder: str | None = None,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    outputs = build_gw1_hybrid_outputs(
        use_cache=use_cache,
        include_unmatched=include_unmatched,
        unmatched_penalty=unmatched_penalty,
        season_folder=season_folder,
        progress_callback=None,
        verbose=verbose,
    )

    candidate_pool = outputs["candidate_pool"]
    hybrid_squad = outputs["hybrid_squad"]
    summary = outputs["summary"]

    print("\nCandidate pool shape:", candidate_pool.shape)
    if not candidate_pool.empty:
        print("Matched players:", int(candidate_pool["matched_previous_season"].sum()))
        print("Unmatched players:", int((candidate_pool["matched_previous_season"] == 0).sum()))
        print("\nPosition counts in candidate pool:")
        print(candidate_pool["position"].value_counts())
        print("\nClub count max in candidate pool:")
        print(candidate_pool["team"].value_counts().max())
        print("\nTop 10 candidate players by hybrid score:")
        print(
            candidate_pool[
                ["name", "position", "team", "price_m", "matched_previous_season", "hybrid_score"]
            ].sort_values("hybrid_score", ascending=False).head(10)
        )

    print("\nHybrid squad:")
    if not hybrid_squad.empty:
        print(
            hybrid_squad[
                ["name", "position", "team", "price_m", "matched_previous_season", "hybrid_score"]
            ].sort_values(["position", "hybrid_score"], ascending=[True, False])
        )

    print("\nSummary:")
    print(summary)

    return outputs