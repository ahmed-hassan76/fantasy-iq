from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import requests

from src.constants import (
    BOOTSTRAP_CACHE_FILE,
    BOOTSTRAP_STATIC_URL,
    ELEMENT_SUMMARY_CACHE_DIR,
    ELEMENT_SUMMARY_URL,
    FIXTURES_CACHE_FILE,
    FIXTURES_URL,
    POSITION_MAP,
    REQUEST_TIMEOUT_SECONDS,
)

ProgressCallback = Callable[[float, str], None]


class FPLApiError(Exception):
    """Raised when an FPL API request fails."""


def _save_json(filepath: Path, payload: Any) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_json(filepath: Path) -> Any:
    with filepath.open("r", encoding="utf-8") as f:
        return json.load(f)


def _emit_progress(progress_callback: ProgressCallback | None, value: float, message: str) -> None:
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, value)), message)


def clear_api_cache() -> None:
    """
    Clear saved on-disk API cache files.
    Useful when the app needs to force a truly fresh live pull.
    """
    if BOOTSTRAP_CACHE_FILE.exists():
        BOOTSTRAP_CACHE_FILE.unlink()

    if FIXTURES_CACHE_FILE.exists():
        FIXTURES_CACHE_FILE.unlink()

    if ELEMENT_SUMMARY_CACHE_DIR.exists():
        for file in ELEMENT_SUMMARY_CACHE_DIR.glob("*.json"):
            try:
                file.unlink()
            except OSError:
                pass


def _get_json(
    url: str,
    cache_path: Path | None = None,
    use_cache: bool = True,
    save_cache: bool = True,
) -> Any:
    """
    Fetch JSON from a live endpoint.
    Falls back to disk cache only if live fetch fails and use_cache=True.
    """
    try:
        response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()

        if save_cache and cache_path is not None:
            _save_json(cache_path, payload)

        return payload

    except (requests.RequestException, ValueError) as exc:
        if use_cache and cache_path is not None and cache_path.exists():
            return _load_json(cache_path)

        raise FPLApiError(f"Failed to fetch data from {url}") from exc


def fetch_bootstrap_static(use_cache: bool = True) -> dict[str, Any]:
    payload = _get_json(
        url=BOOTSTRAP_STATIC_URL,
        cache_path=BOOTSTRAP_CACHE_FILE,
        use_cache=use_cache,
        save_cache=True,
    )
    if not isinstance(payload, dict):
        raise FPLApiError("bootstrap-static response is not a JSON object.")
    return payload


def fetch_fixtures(use_cache: bool = True) -> list[dict[str, Any]]:
    payload = _get_json(
        url=FIXTURES_URL,
        cache_path=FIXTURES_CACHE_FILE,
        use_cache=use_cache,
        save_cache=True,
    )
    if not isinstance(payload, list):
        raise FPLApiError("fixtures response is not a JSON list.")
    return payload


def fetch_element_summary(player_id: int, use_cache: bool = True) -> dict[str, Any]:
    cache_path = ELEMENT_SUMMARY_CACHE_DIR / f"{player_id}.json"
    payload = _get_json(
        url=ELEMENT_SUMMARY_URL.format(player_id=player_id),
        cache_path=cache_path,
        use_cache=use_cache,
        save_cache=True,
    )
    if not isinstance(payload, dict):
        raise FPLApiError(f"element-summary response for player {player_id} is not a JSON object.")
    return payload


def get_players_df(use_cache: bool = True) -> pd.DataFrame:
    bootstrap = fetch_bootstrap_static(use_cache=use_cache)
    players = pd.DataFrame(bootstrap["elements"])

    if players.empty:
        return players

    players["position"] = players["element_type"].map(POSITION_MAP)

    # Correct FPL price conversion: now_cost is in tenths
    players["price_m"] = pd.to_numeric(players["now_cost"], errors="coerce") / 10.0

    players["full_name"] = (
        players["first_name"].fillna("").astype(str).str.strip()
        + " "
        + players["second_name"].fillna("").astype(str).str.strip()
    ).str.strip()

    return players


def get_teams_df(use_cache: bool = True) -> pd.DataFrame:
    bootstrap = fetch_bootstrap_static(use_cache=use_cache)
    teams = pd.DataFrame(bootstrap["teams"])

    if teams.empty:
        return teams

    teams = teams.rename(
        columns={
            "id": "team_id",
            "name": "team_name",
            "short_name": "team_short_name",
        }
    )

    return teams


def get_events_df(use_cache: bool = True) -> pd.DataFrame:
    bootstrap = fetch_bootstrap_static(use_cache=use_cache)
    events = pd.DataFrame(bootstrap["events"])

    if events.empty:
        return events

    events = events.rename(columns={"id": "event_id", "name": "event_name"})
    return events


def get_fixtures_df(use_cache: bool = True) -> pd.DataFrame:
    fixtures = fetch_fixtures(use_cache=use_cache)
    fixtures_df = pd.DataFrame(fixtures)

    if fixtures_df.empty:
        return fixtures_df

    return fixtures_df


def get_player_history_df(player_id: int, use_cache: bool = True) -> pd.DataFrame:
    payload = fetch_element_summary(player_id=player_id, use_cache=use_cache)
    history_df = pd.DataFrame(payload.get("history", []))

    if history_df.empty:
        return history_df

    history_df["player_id"] = player_id
    return history_df


def get_player_fixtures_df(player_id: int, use_cache: bool = True) -> pd.DataFrame:
    payload = fetch_element_summary(player_id=player_id, use_cache=use_cache)
    fixtures_df = pd.DataFrame(payload.get("fixtures", []))

    if fixtures_df.empty:
        return fixtures_df

    fixtures_df["player_id"] = player_id
    return fixtures_df


def get_all_player_histories_df(
    player_ids: list[int],
    use_cache: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    history_frames: list[pd.DataFrame] = []
    total_players = len(player_ids)

    _emit_progress(progress_callback, 0.0, "Starting player history fetch...")

    for idx, player_id in enumerate(player_ids, start=1):
        try:
            player_history = get_player_history_df(player_id=player_id, use_cache=use_cache)
            if not player_history.empty:
                history_frames.append(player_history)
        except FPLApiError:
            pass

        if idx % 25 == 0 or idx == total_players:
            pct = idx / total_players if total_players > 0 else 1.0
            _emit_progress(
                progress_callback,
                pct,
                f"Fetched player histories: {idx}/{total_players}",
            )

    if not history_frames:
        return pd.DataFrame()

    _emit_progress(progress_callback, 1.0, "Player history fetch complete.")
    return pd.concat(history_frames, ignore_index=True)


def get_all_player_upcoming_fixtures_df(
    player_ids: list[int],
    use_cache: bool = True,
) -> pd.DataFrame:
    fixture_frames: list[pd.DataFrame] = []

    for player_id in player_ids:
        try:
            player_fixtures = get_player_fixtures_df(player_id=player_id, use_cache=use_cache)
            if not player_fixtures.empty:
                fixture_frames.append(player_fixtures)
        except FPLApiError:
            continue

    if not fixture_frames:
        return pd.DataFrame()

    return pd.concat(fixture_frames, ignore_index=True)


def get_base_fpl_tables(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    players_df = get_players_df(use_cache=use_cache)
    teams_df = get_teams_df(use_cache=use_cache)
    events_df = get_events_df(use_cache=use_cache)
    fixtures_df = get_fixtures_df(use_cache=use_cache)

    return {
        "players": players_df,
        "teams": teams_df,
        "events": events_df,
        "fixtures": fixtures_df,
    }