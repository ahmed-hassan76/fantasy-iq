from __future__ import annotations

from itertools import combinations
from typing import Iterable

import pandas as pd

from src.predict import build_predictions_table


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def validate_full_squad(
    squad_df: pd.DataFrame,
    money_in_bank: float = 0.0,
    budget_limit: float = 100.0,
    enforce_budget: bool = False,
) -> tuple[bool, list[str], float, float]:
    """
    Validate a full 15-player FPL squad.

    For current-season squads in the Transfer Assistant, enforce_budget should usually be False,
    because squad market value can exceed the original 100.0m due to price changes.
    """
    reasons: list[str] = []

    counts = squad_df["position"].value_counts()

    total_players = len(squad_df)
    gk_count = counts.get("GK", 0)
    def_count = counts.get("DEF", 0)
    mid_count = counts.get("MID", 0)
    fwd_count = counts.get("FWD", 0)

    squad_cost = round(float(squad_df["price_m"].sum()), 1)
    total_budget_used = round(float(squad_cost + money_in_bank), 1)

    max_club_players = (
        int(squad_df["team"].value_counts().max()) if len(squad_df) > 0 else 0
    )

    if total_players != 15:
        reasons.append(f"You selected {total_players} players. Exactly 15 are required.")

    if gk_count != 2:
        reasons.append(f"You selected {gk_count} goalkeepers. Exactly 2 are required.")

    if def_count != 5:
        reasons.append(f"You selected {def_count} defenders. Exactly 5 are required.")

    if mid_count != 5:
        reasons.append(f"You selected {mid_count} midfielders. Exactly 5 are required.")

    if fwd_count != 3:
        reasons.append(f"You selected {fwd_count} forwards. Exactly 3 are required.")

    if max_club_players > 3:
        reasons.append(
            f"You selected {max_club_players} players from the same club. Maximum allowed is 3."
        )

    if enforce_budget and total_budget_used > budget_limit + 1e-9:
        reasons.append(
            f"Squad cost plus money in bank is {total_budget_used:.1f}. "
            f"Maximum allowed budget is {budget_limit:.1f}."
        )

    valid = len(reasons) == 0
    return valid, reasons, squad_cost, total_budget_used


def validate_starting_xi(starting_df: pd.DataFrame) -> bool:
    """
    Validate a starting XI under standard FPL formation rules.
    """
    counts = starting_df["position"].value_counts()

    total_players = len(starting_df)
    gk_count = counts.get("GK", 0)
    def_count = counts.get("DEF", 0)
    mid_count = counts.get("MID", 0)
    fwd_count = counts.get("FWD", 0)

    valid = (
        total_players == 11
        and gk_count == 1
        and 3 <= def_count <= 5
        and 2 <= mid_count <= 5
        and 1 <= fwd_count <= 3
    )

    return valid


def _club_limit_ok(team_series: pd.Series, club_limit: int = 3) -> bool:
    """
    Check if no team exceeds the club limit.
    """
    if team_series.empty:
        return True
    return int(team_series.value_counts().max()) <= club_limit


def recommend_best_one_transfer(
    current_squad_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    money_in_bank: float = 0.0,
    starting_names: Iterable[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Recommend the best single transfer based on predicted point gain.
    """
    if starting_names is None:
        starting_names = []

    _log("[1-transfer] Validating current squad...", verbose)
    valid, reasons, squad_cost, total_budget_used = validate_full_squad(
        current_squad_df, money_in_bank=money_in_bank
    )

    if not valid:
        raise ValueError(f"Current squad is invalid: {reasons}")

    _log(
        f"[1-transfer] Squad valid. Squad cost = {squad_cost:.1f}, total budget used = {total_budget_used:.1f}",
        verbose,
    )

    current_names = set(current_squad_df["name"].tolist())
    current_club_counts = current_squad_df["team"].value_counts().to_dict()

    recommendations: list[dict] = []

    total_outgoing = len(current_squad_df)
    for idx, (_, outgoing) in enumerate(current_squad_df.iterrows(), start=1):
        if idx % 5 == 0 or idx == total_outgoing:
            _log(f"[1-transfer] Outgoing progress: {idx}/{total_outgoing}", verbose)

        outgoing_name = outgoing["name"]
        outgoing_position = outgoing["position"]
        outgoing_team = outgoing["team"]
        outgoing_price = float(outgoing["price_m"])
        outgoing_pred = float(outgoing["predicted_points"])

        max_affordable_price = outgoing_price + money_in_bank

        candidate_incomings = predictions_df[
            (predictions_df["position"] == outgoing_position)
            & (~predictions_df["name"].isin(current_names))
            & (predictions_df["price_m"] <= max_affordable_price)
        ].copy()

        for _, incoming in candidate_incomings.iterrows():
            incoming_name = incoming["name"]
            incoming_team = incoming["team"]
            incoming_price = float(incoming["price_m"])
            incoming_pred = float(incoming["predicted_points"])

            new_club_counts = current_club_counts.copy()
            new_club_counts[outgoing_team] -= 1
            if new_club_counts[outgoing_team] == 0:
                del new_club_counts[outgoing_team]

            new_club_counts[incoming_team] = new_club_counts.get(incoming_team, 0) + 1

            if max(new_club_counts.values()) > 3:
                continue

            points_gain = incoming_pred - outgoing_pred
            budget_change = incoming_price - outgoing_price
            remaining_bank = money_in_bank - budget_change

            recommendations.append({
                "player_out": outgoing_name,
                "player_out_team": outgoing_team,
                "player_out_position": outgoing_position,
                "player_out_price": outgoing_price,
                "player_out_predicted_points": outgoing_pred,

                "player_in": incoming_name,
                "player_in_team": incoming_team,
                "player_in_position": outgoing_position,
                "player_in_price": incoming_price,
                "player_in_predicted_points": incoming_pred,

                "predicted_points_gain": points_gain,
                "budget_change": budget_change,
                "remaining_money_in_bank": remaining_bank,
                "outgoing_is_starter": outgoing_name in starting_names,
            })

    if len(recommendations) == 0:
        _log("[1-transfer] No valid transfer recommendations found.", verbose)
        return pd.DataFrame()

    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df = recommendations_df.sort_values(
        ["predicted_points_gain", "player_in_predicted_points"],
        ascending=[False, False],
    ).reset_index(drop=True)

    _log(f"[1-transfer] Recommendations found: {len(recommendations_df)}", verbose)
    return recommendations_df


def recommend_best_two_transfers(
    current_squad_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    money_in_bank: float = 0.0,
    starting_names: Iterable[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Recommend the best two-transfer combination.
    """
    if starting_names is None:
        starting_names = []

    _log("[2-transfer] Validating current squad...", verbose)
    valid, reasons, squad_cost, total_budget_used = validate_full_squad(
        current_squad_df, money_in_bank=money_in_bank
    )

    if not valid:
        raise ValueError(f"Current squad is invalid: {reasons}")

    _log(
        f"[2-transfer] Squad valid. Squad cost = {squad_cost:.1f}, total budget used = {total_budget_used:.1f}",
        verbose,
    )

    current_names = set(current_squad_df["name"].tolist())
    current_club_counts = current_squad_df["team"].value_counts().to_dict()

    recommendations: list[dict] = []
    current_squad_list = current_squad_df.to_dict("records")

    outgoing_pairs = list(combinations(range(len(current_squad_list)), 2))
    total_pairs = len(outgoing_pairs)

    for pair_idx, (i, j) in enumerate(outgoing_pairs, start=1):
        if pair_idx % 50 == 0 or pair_idx == total_pairs:
            _log(f"[2-transfer] Outgoing pair progress: {pair_idx}/{total_pairs}", verbose)

        outgoing1 = current_squad_list[i]
        outgoing2 = current_squad_list[j]

        out1_name = outgoing1["name"]
        out2_name = outgoing2["name"]

        out1_pos = outgoing1["position"]
        out2_pos = outgoing2["position"]

        out1_team = outgoing1["team"]
        out2_team = outgoing2["team"]

        out1_price = float(outgoing1["price_m"])
        out2_price = float(outgoing2["price_m"])

        out1_pred = float(outgoing1["predicted_points"])
        out2_pred = float(outgoing2["predicted_points"])

        max_affordable_price = out1_price + out2_price + money_in_bank

        incoming_pool1 = predictions_df[
            (predictions_df["position"] == out1_pos)
            & (~predictions_df["name"].isin(current_names))
        ].copy()

        incoming_pool2 = predictions_df[
            (predictions_df["position"] == out2_pos)
            & (~predictions_df["name"].isin(current_names))
        ].copy()

        incoming_list1 = incoming_pool1.to_dict("records")
        incoming_list2 = incoming_pool2.to_dict("records")

        for in1 in incoming_list1:
            for in2 in incoming_list2:
                in1_name = in1["name"]
                in2_name = in2["name"]

                if in1_name == in2_name:
                    continue

                total_in_price = float(in1["price_m"]) + float(in2["price_m"])
                if total_in_price > max_affordable_price:
                    continue

                new_club_counts = current_club_counts.copy()

                new_club_counts[out1_team] -= 1
                if new_club_counts[out1_team] == 0:
                    del new_club_counts[out1_team]

                new_club_counts[out2_team] -= 1
                if out2_team in new_club_counts and new_club_counts[out2_team] == 0:
                    del new_club_counts[out2_team]

                new_club_counts[in1["team"]] = new_club_counts.get(in1["team"], 0) + 1
                new_club_counts[in2["team"]] = new_club_counts.get(in2["team"], 0) + 1

                if max(new_club_counts.values()) > 3:
                    continue

                points_gain = (
                    float(in1["predicted_points"]) + float(in2["predicted_points"])
                    - (out1_pred + out2_pred)
                )
                budget_change = total_in_price - (out1_price + out2_price)
                remaining_bank = money_in_bank - budget_change

                recommendations.append({
                    "player_out_1": out1_name,
                    "player_out_2": out2_name,
                    "player_out_positions": f"{out1_pos}, {out2_pos}",

                    "player_in_1": in1_name,
                    "player_in_2": in2_name,
                    "player_in_positions": f"{in1['position']}, {in2['position']}",

                    "predicted_points_gain": points_gain,
                    "budget_change": budget_change,
                    "remaining_money_in_bank": remaining_bank,

                    "outgoing_1_is_starter": out1_name in starting_names,
                    "outgoing_2_is_starter": out2_name in starting_names,
                })

    if len(recommendations) == 0:
        _log("[2-transfer] No valid transfer recommendations found.", verbose)
        return pd.DataFrame()

    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df = recommendations_df.sort_values(
        ["predicted_points_gain"],
        ascending=[False],
    ).reset_index(drop=True)

    _log(f"[2-transfer] Recommendations found: {len(recommendations_df)}", verbose)
    return recommendations_df


def build_current_squad_from_names(
    selected_names: list[str],
    predictions_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a current squad table from a list of selected player names.
    """
    squad_df = predictions_df[predictions_df["name"].isin(selected_names)].copy()

    if len(squad_df) != len(selected_names):
        found_names = set(squad_df["name"].tolist())
        missing_names = [name for name in selected_names if name not in found_names]
        raise ValueError(f"Some selected players were not found in predictions: {missing_names}")

    return squad_df.reset_index(drop=True)


def inspect_transfer_logic(
    current_squad_names: list[str],
    money_in_bank: float = 0.0,
    starting_names: list[str] | None = None,
    use_cache: bool = True,
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Helper for testing transfer logic against a supplied current squad.
    """
    if starting_names is None:
        starting_names = []

    _log("[Transfer Logic] Building latest predictions...", verbose)
    predictions_df = build_predictions_table(use_cache=use_cache, verbose=verbose)

    _log("[Transfer Logic] Building current squad from supplied names...", verbose)
    current_squad_df = build_current_squad_from_names(
        selected_names=current_squad_names,
        predictions_df=predictions_df,
    )

    valid, reasons, squad_cost, total_budget_used = validate_full_squad(
        current_squad_df,
        money_in_bank=money_in_bank,
    )

    print("\nCurrent squad valid:", valid)
    print("Validation reasons:", reasons)
    print("Squad cost:", squad_cost)
    print("Total budget used:", total_budget_used)

    one_transfer_df = recommend_best_one_transfer(
        current_squad_df=current_squad_df,
        predictions_df=predictions_df,
        money_in_bank=money_in_bank,
        starting_names=starting_names,
        verbose=verbose,
    )

    two_transfer_df = recommend_best_two_transfers(
        current_squad_df=current_squad_df,
        predictions_df=predictions_df,
        money_in_bank=money_in_bank,
        starting_names=starting_names,
        verbose=verbose,
    )

    print("\nTop 5 one-transfer recommendations:")
    if not one_transfer_df.empty:
        print(one_transfer_df.head(5))
    else:
        print("No one-transfer recommendations found.")

    print("\nTop 5 two-transfer recommendations:")
    if not two_transfer_df.empty:
        print(two_transfer_df.head(5))
    else:
        print("No two-transfer recommendations found.")

    return {
        "predictions_df": predictions_df,
        "current_squad_df": current_squad_df,
        "one_transfer_df": one_transfer_df,
        "two_transfer_df": two_transfer_df,
    }