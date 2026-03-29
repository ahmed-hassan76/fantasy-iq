from __future__ import annotations

import pandas as pd
import pulp

from src.predict import build_predictions_table


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def validate_prediction_table(df: pd.DataFrame) -> None:
    required_cols = [
        "name",
        "team",
        "position",
        "price_m",
        "predicted_points",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Prediction table is missing required columns: {missing_cols}")


def optimize_best_15_squad(
    predictions_df: pd.DataFrame,
    budget_limit: float = 100.0,
    club_limit: int = 3,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build the best full 15-player squad using predicted points.
    """
    _log("[1/5] Validating prediction table...", verbose)
    validate_prediction_table(predictions_df)

    candidate_df = predictions_df.copy().reset_index(drop=True)

    _log(f"Candidate players available: {len(candidate_df)}", verbose)

    _log("[2/5] Building optimization problem...", verbose)
    problem = pulp.LpProblem("Best_15_Player_Squad", pulp.LpMaximize)

    player_vars = {
        i: pulp.LpVariable(f"player_{i}", cat="Binary")
        for i in candidate_df.index
    }

    # Objective: maximize predicted points
    problem += pulp.lpSum(
        candidate_df.loc[i, "predicted_points"] * player_vars[i]
        for i in candidate_df.index
    )

    # Squad size
    problem += pulp.lpSum(player_vars[i] for i in candidate_df.index) == 15

    # Position constraints
    problem += pulp.lpSum(
        player_vars[i]
        for i in candidate_df.index
        if candidate_df.loc[i, "position"] == "GK"
    ) == 2

    problem += pulp.lpSum(
        player_vars[i]
        for i in candidate_df.index
        if candidate_df.loc[i, "position"] == "DEF"
    ) == 5

    problem += pulp.lpSum(
        player_vars[i]
        for i in candidate_df.index
        if candidate_df.loc[i, "position"] == "MID"
    ) == 5

    problem += pulp.lpSum(
        player_vars[i]
        for i in candidate_df.index
        if candidate_df.loc[i, "position"] == "FWD"
    ) == 3

    # Budget
    problem += pulp.lpSum(
        candidate_df.loc[i, "price_m"] * player_vars[i]
        for i in candidate_df.index
    ) <= budget_limit

    # Club limit
    unique_teams = candidate_df["team"].dropna().unique().tolist()
    for idx, team in enumerate(unique_teams, start=1):
        if idx % 5 == 0 or idx == len(unique_teams):
            _log(f"  Club constraint progress: {idx}/{len(unique_teams)}", verbose)

        problem += pulp.lpSum(
            player_vars[i]
            for i in candidate_df.index
            if candidate_df.loc[i, "team"] == team
        ) <= club_limit

    _log("[3/5] Solving optimization problem...", verbose)
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    status = pulp.LpStatus[problem.status]
    _log(f"Solver status: {status}", verbose)

    if status != "Optimal":
        raise ValueError(f"Optimization did not return an optimal solution. Status: {status}")

    _log("[4/5] Extracting selected squad...", verbose)
    selected_indices = [
        i for i in candidate_df.index
        if player_vars[i].value() == 1
    ]

    squad_df = candidate_df.loc[selected_indices].copy().reset_index(drop=True)

    _log("[5/5] Final squad built.", verbose)
    _log(f"Selected players: {len(squad_df)}", verbose)
    _log(f"Total predicted points: {squad_df['predicted_points'].sum():.2f}", verbose)
    _log(f"Total cost: {squad_df['price_m'].sum():.1f}", verbose)

    return squad_df


def summarize_squad(squad_df: pd.DataFrame) -> dict:
    """
    Return a simple summary of the optimized squad.
    """
    if squad_df.empty:
        return {
            "players": 0,
            "total_cost": 0.0,
            "total_predicted_points": 0.0,
            "position_counts": {},
            "club_counts": {},
        }

    summary = {
        "players": len(squad_df),
        "total_cost": float(squad_df["price_m"].sum()),
        "total_predicted_points": float(squad_df["predicted_points"].sum()),
        "position_counts": squad_df["position"].value_counts().to_dict(),
        "club_counts": squad_df["team"].value_counts().to_dict(),
    }
    return summary

def build_optimized_squad_from_predictions(
    predictions_df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Build optimized squad directly from an already prepared predictions table.
    This avoids rerunning the full live prediction pipeline.
    """
    squad_df = optimize_best_15_squad(
        predictions_df=predictions_df,
        budget_limit=100.0,
        club_limit=3,
        verbose=verbose,
    )
    return squad_df

def build_optimized_squad(use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Convenience wrapper:
    - build live predictions
    - optimize best 15-man squad
    """
    _log("Building live predictions for squad optimization...", verbose)
    predictions_df = build_predictions_table(use_cache=use_cache, verbose=verbose)

    squad_df = optimize_best_15_squad(
        predictions_df=predictions_df,
        budget_limit=100.0,
        club_limit=3,
        verbose=verbose,
    )

    return squad_df


def inspect_optimized_squad(use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    squad_df = build_optimized_squad(use_cache=use_cache, verbose=verbose)
    summary = summarize_squad(squad_df)

    print("\nOptimized squad shape:", squad_df.shape)
    print("\nSquad summary:")
    print(summary)

    print("\nSquad table:")
    print(
        squad_df[
            ["name", "position", "team", "price_m", "predicted_points", "model_used"]
        ].sort_values(["position", "predicted_points"], ascending=[True, False])
    )

    return squad_df