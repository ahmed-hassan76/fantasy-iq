from __future__ import annotations

import pandas as pd

from src.api import get_fixtures_df, get_teams_df


SUMMARY_WINDOWS = (3, 5)


def _coerce_bool_series(series: pd.Series) -> pd.Series:
    if series.empty:
        return pd.Series(dtype=bool)

    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    normalized = series.astype("string").str.lower().str.strip()
    return normalized.isin({"true", "1", "yes"})


def _format_fixture_label(row: pd.Series) -> str:
    opponent = row.get("opponent_short_name")
    if pd.isna(opponent) or str(opponent).strip() == "":
        opponent = "TBC"

    venue = "H" if bool(row.get("is_home")) else "A"

    difficulty = row.get("difficulty")
    if pd.isna(difficulty):
        difficulty_text = "?"
    else:
        difficulty_text = str(int(difficulty))

    return f"{opponent} ({venue}) FDR {difficulty_text}"


def build_normalized_team_fixtures(use_cache: bool = True) -> pd.DataFrame:
    """
    Return one row per team fixture from the official FPL fixtures endpoint.
    Double gameweeks naturally produce multiple rows for the same team/event.
    """
    try:
        fixtures = get_fixtures_df(use_cache=use_cache).copy()
    except Exception:
        return pd.DataFrame()

    try:
        teams = get_teams_df(use_cache=use_cache).copy()
    except Exception:
        teams = pd.DataFrame()

    if fixtures.empty:
        return pd.DataFrame()

    required_cols = [
        "id",
        "event",
        "kickoff_time",
        "team_h",
        "team_a",
        "team_h_difficulty",
        "team_a_difficulty",
        "finished",
        "started",
    ]
    for col in required_cols:
        if col not in fixtures.columns:
            fixtures[col] = pd.NA

    fixtures["fixture_id"] = pd.to_numeric(fixtures["id"], errors="coerce")
    fixtures["event"] = pd.to_numeric(fixtures["event"], errors="coerce")
    fixtures["team_h"] = pd.to_numeric(fixtures["team_h"], errors="coerce")
    fixtures["team_a"] = pd.to_numeric(fixtures["team_a"], errors="coerce")
    fixtures["team_h_difficulty"] = pd.to_numeric(fixtures["team_h_difficulty"], errors="coerce")
    fixtures["team_a_difficulty"] = pd.to_numeric(fixtures["team_a_difficulty"], errors="coerce")
    fixtures["kickoff_time"] = pd.to_datetime(fixtures["kickoff_time"], errors="coerce", utc=True)

    finished = _coerce_bool_series(fixtures["finished"])
    started = _coerce_bool_series(fixtures["started"])
    future_fixtures = fixtures[~finished & ~started].copy()

    if future_fixtures.empty:
        return pd.DataFrame()

    home_rows = pd.DataFrame(
        {
            "team_id": future_fixtures["team_h"],
            "event": future_fixtures["event"],
            "fixture_id": future_fixtures["fixture_id"],
            "opponent_team_id": future_fixtures["team_a"],
            "is_home": True,
            "difficulty": future_fixtures["team_h_difficulty"],
            "kickoff_time": future_fixtures["kickoff_time"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "team_id": future_fixtures["team_a"],
            "event": future_fixtures["event"],
            "fixture_id": future_fixtures["fixture_id"],
            "opponent_team_id": future_fixtures["team_h"],
            "is_home": False,
            "difficulty": future_fixtures["team_a_difficulty"],
            "kickoff_time": future_fixtures["kickoff_time"],
        }
    )

    normalized = pd.concat([home_rows, away_rows], ignore_index=True)
    normalized = normalized.dropna(subset=["team_id"]).copy()

    if normalized.empty:
        return pd.DataFrame()

    normalized["team_id"] = pd.to_numeric(normalized["team_id"], errors="coerce")
    normalized["opponent_team_id"] = pd.to_numeric(normalized["opponent_team_id"], errors="coerce")
    normalized["difficulty"] = pd.to_numeric(normalized["difficulty"], errors="coerce")

    if not teams.empty:
        team_lookup_cols = [col for col in ["team_id", "team_name", "team_short_name"] if col in teams.columns]
        if "team_id" in team_lookup_cols:
            team_lookup = teams[team_lookup_cols].copy()
            team_lookup["team_id"] = pd.to_numeric(team_lookup["team_id"], errors="coerce")

            normalized = normalized.merge(team_lookup, on="team_id", how="left")
            opponent_lookup = team_lookup.rename(
                columns={
                    "team_id": "opponent_team_id",
                    "team_name": "opponent_team_name",
                    "team_short_name": "opponent_short_name",
                }
            )
            normalized = normalized.merge(opponent_lookup, on="opponent_team_id", how="left")

    for col in ["team_name", "team_short_name", "opponent_short_name"]:
        if col not in normalized.columns:
            normalized[col] = pd.NA

    normalized["fixture_label"] = normalized.apply(_format_fixture_label, axis=1)

    return normalized.sort_values(
        ["event", "kickoff_time", "fixture_id"],
        na_position="last",
    ).reset_index(drop=True)


def build_team_fixture_summary(use_cache: bool = True) -> pd.DataFrame:
    fixtures = build_normalized_team_fixtures(use_cache=use_cache)

    if fixtures.empty or "team_id" not in fixtures.columns:
        return pd.DataFrame()

    summary_rows: list[dict[str, object]] = []

    for team_id, team_fixtures in fixtures.groupby("team_id", dropna=True):
        team_fixtures = team_fixtures.sort_values(
            ["event", "kickoff_time", "fixture_id"],
            na_position="last",
        ).reset_index(drop=True)

        row: dict[str, object] = {"team_id": team_id}

        if "team_name" in team_fixtures.columns:
            row["team_name"] = team_fixtures["team_name"].dropna().iloc[0] if team_fixtures["team_name"].notna().any() else pd.NA
        if "team_short_name" in team_fixtures.columns:
            row["team_short_name"] = team_fixtures["team_short_name"].dropna().iloc[0] if team_fixtures["team_short_name"].notna().any() else pd.NA

        for window in SUMMARY_WINDOWS:
            window_df = team_fixtures.head(window)
            row[f"next_{window}_fixtures"] = ", ".join(window_df["fixture_label"].dropna().astype(str).tolist())
            row[f"next_{window}_fdr_avg"] = pd.to_numeric(window_df["difficulty"], errors="coerce").mean()
            row[f"next_{window}_fixture_count"] = int(len(window_df))

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def add_fixture_summary_to_predictions(
    predictions_df: pd.DataFrame,
    use_cache: bool = True,
) -> pd.DataFrame:
    if predictions_df.empty:
        return predictions_df.copy()

    output_df = predictions_df.copy()
    summary_df = build_team_fixture_summary(use_cache=use_cache)

    if summary_df.empty:
        return output_df

    if "team_id" in output_df.columns and "team_id" in summary_df.columns:
        output_df["team_id"] = pd.to_numeric(output_df["team_id"], errors="coerce")
        summary_df["team_id"] = pd.to_numeric(summary_df["team_id"], errors="coerce")
        if output_df["team_id"].notna().any():
            return output_df.merge(
                summary_df.drop(columns=["team_name", "team_short_name"], errors="ignore"),
                on="team_id",
                how="left",
            )

    if "team" in output_df.columns and "team_name" in summary_df.columns:
        return output_df.merge(
            summary_df.drop(columns=["team_id", "team_short_name"], errors="ignore"),
            left_on="team",
            right_on="team_name",
            how="left",
        ).drop(columns=["team_name"], errors="ignore")

    return output_df
