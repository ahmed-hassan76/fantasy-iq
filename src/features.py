from __future__ import annotations

from typing import Callable
import pandas as pd

from src.constants import DEF_FEATURES, FWD_FEATURES, GK_FEATURES, MID_FEATURES
from src.preprocess import build_clean_history_table

ProgressCallback = Callable[[float, str], None]


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def _emit_progress(progress_callback: ProgressCallback | None, value: float, message: str) -> None:
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, value)), message)


def build_feature_table(
    use_cache: bool = True,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
    drop_missing_target: bool = True,
) -> pd.DataFrame:
    _log("[1/8] Loading clean history table...", verbose)

    def preprocess_progress(local_progress: float, message: str) -> None:
        mapped_progress = 0.00 + (0.50 * local_progress)
        _emit_progress(progress_callback, mapped_progress, message)

    df = build_clean_history_table(
        use_cache=use_cache,
        progress_callback=preprocess_progress,
    )

    if df.empty:
        _log("No data returned from preprocessing.", verbose)
        return pd.DataFrame()

    _log(f"Loaded clean history table with {len(df)} rows.", verbose)
    _emit_progress(progress_callback, 0.55, "Sorting clean history table...")

    df = df.sort_values(["name", "round"]).reset_index(drop=True)

    _emit_progress(progress_callback, 0.60, "Aggregating duplicate player-round rows...")
    df = df.groupby(
        ["name", "position", "round"],
        as_index=False
    ).agg({
        "assists": "sum",
        "clean_sheets": "sum",
        "goals_conceded": "sum",
        "goals_scored": "sum",
        "minutes": "sum",
        "saves": "sum",
        "starts": "sum",
        "value": "first",
        "was_home": "first",
        "total_points": "sum",
        "team": "first",
        "status": "first",
        "chance_of_playing_next_round": "first",
    })

    lag_features = [
        "total_points",
        "goals_scored",
        "assists",
        "clean_sheets",
        "saves",
        "goals_conceded",
        "minutes",
    ]

    _emit_progress(progress_callback, 0.68, "Creating lag features...")
    total_lag = len(lag_features)
    for i, col in enumerate(lag_features, start=1):
        df[f"{col}_lag1"] = df.groupby("name")[col].shift(1)
        _log(f"  Lag feature {i}/{total_lag} complete: {col}_lag1", verbose)

    rolling_features = [
        "total_points",
        "goals_scored",
        "assists",
        "minutes",
    ]

    _emit_progress(progress_callback, 0.78, "Creating rolling features...")
    total_rolling = len(rolling_features)
    for i, col in enumerate(rolling_features, start=1):
        df[f"{col}_rolling3"] = (
            df.groupby("name")[col]
            .rolling(window=3)
            .mean()
            .reset_index(level=0, drop=True)
        )
        _log(f"  Rolling feature {i}/{total_rolling} complete: {col}_rolling3", verbose)

    _emit_progress(progress_callback, 0.88, "Creating consistency and availability features...")
    df["points_std3"] = (
        df.groupby("name")["total_points"]
        .rolling(window=3)
        .std()
        .reset_index(level=0, drop=True)
    )
    df["played_last_gw"] = (df["minutes_lag1"] > 0).astype(int)

    _emit_progress(progress_callback, 0.94, "Creating training target and removing leakage...")
    df["target"] = df.groupby("name")["total_points"].shift(-1)
    df = df.drop(columns=["total_points"])

    before_drop = len(df)
    if drop_missing_target:
        df = df.dropna().reset_index(drop=True)
    else:
        required_feature_cols = sorted(set(GK_FEATURES + DEF_FEATURES + MID_FEATURES + FWD_FEATURES))
        df = df.dropna(subset=required_feature_cols).reset_index(drop=True)
    after_drop = len(df)
    _log(f"Dropped {before_drop - after_drop} rows with incomplete history/target.", verbose)

    df["was_home"] = df["was_home"].astype(int)
    df["played_last_gw"] = df["played_last_gw"].astype(int)

    _emit_progress(progress_callback, 1.0, "Feature engineering complete.")
    return df


def split_position_datasets(feature_df: pd.DataFrame, verbose: bool = True) -> dict[str, pd.DataFrame]:
    _log("Splitting feature table by position...", verbose)

    if feature_df.empty:
        _log("Feature table is empty. Returning empty datasets.", verbose)
        return {"GK": pd.DataFrame(), "DEF": pd.DataFrame(), "MID": pd.DataFrame(), "FWD": pd.DataFrame()}

    gk_df = feature_df[feature_df["position"] == "GK"].copy()
    def_df = feature_df[feature_df["position"] == "DEF"].copy()
    mid_df = feature_df[feature_df["position"] == "MID"].copy()
    fwd_df = feature_df[feature_df["position"] == "FWD"].copy()

    passthrough_cols = [
        "status",
        "chance_of_playing_next_round",
        "starts",
        "minutes_lag1",
        "minutes_rolling3",
    ]
    passthrough_cols = [col for col in passthrough_cols if col in feature_df.columns]

    def ordered_existing_cols(cols: list[str]) -> list[str]:
        return list(dict.fromkeys(col for col in cols if col in feature_df.columns))

    gk_cols = ordered_existing_cols(["name", "team", "round"] + GK_FEATURES + passthrough_cols + ["target"])
    def_cols = ordered_existing_cols(["name", "team", "round"] + DEF_FEATURES + passthrough_cols + ["target"])
    mid_cols = ordered_existing_cols(["name", "team", "round"] + MID_FEATURES + passthrough_cols + ["target"])
    fwd_cols = ordered_existing_cols(["name", "team", "round"] + FWD_FEATURES + passthrough_cols + ["target"])

    return {
        "GK": gk_df[gk_cols].copy(),
        "DEF": def_df[def_cols].copy(),
        "MID": mid_df[mid_cols].copy(),
        "FWD": fwd_df[fwd_cols].copy(),
    }


def get_latest_round_inference_tables(
    use_cache: bool = True,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, pd.DataFrame]:
    _log("Building latest-round inference tables...", verbose)

    def feature_progress(local_progress: float, message: str) -> None:
        mapped_progress = 0.00 + (0.82 * local_progress)
        _emit_progress(progress_callback, mapped_progress, message)

    feature_df = build_feature_table(
        use_cache=use_cache,
        verbose=verbose,
        progress_callback=feature_progress,
        drop_missing_target=False,
    )

    if feature_df.empty:
        return {"GK": pd.DataFrame(), "DEF": pd.DataFrame(), "MID": pd.DataFrame(), "FWD": pd.DataFrame()}

    _emit_progress(progress_callback, 0.88, "Selecting latest round rows for each player...")
    latest_df = (
        feature_df.sort_values(["name", "round"])
        .groupby("name", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )

    _emit_progress(progress_callback, 0.94, "Splitting inference tables by position...")
    split_tables = split_position_datasets(latest_df, verbose=verbose)

    for position, df in split_tables.items():
        if not df.empty and "target" in df.columns:
            split_tables[position] = df.drop(columns=["target"]).copy()

    _emit_progress(progress_callback, 1.0, "Inference tables ready.")
    return split_tables


def inspect_feature_pipeline(use_cache: bool = True, verbose: bool = True) -> dict[str, pd.DataFrame]:
    feature_df = build_feature_table(use_cache=use_cache, verbose=verbose)
    split_tables = split_position_datasets(feature_df, verbose=verbose)
    inference_tables = get_latest_round_inference_tables(use_cache=use_cache, verbose=verbose)

    return {
        "full_feature_df": feature_df,
        "gk_train": split_tables["GK"],
        "def_train": split_tables["DEF"],
        "mid_train": split_tables["MID"],
        "fwd_train": split_tables["FWD"],
        "gk_infer": inference_tables["GK"],
        "def_infer": inference_tables["DEF"],
        "mid_infer": inference_tables["MID"],
        "fwd_infer": inference_tables["FWD"],
    }
