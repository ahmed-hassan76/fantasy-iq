from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.constants import FEATURES_BY_POSITION, MODEL_PATHS
from src.features import get_latest_round_inference_tables

ProgressCallback = Callable[[float, str], None]


def _log(message: str, verbose: bool = True) -> None:
    if verbose:
        print(message)


def _emit_progress(progress_callback: ProgressCallback | None, value: float, message: str) -> None:
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, value)), message)


def load_position_model(position: str, verbose: bool = True) -> Any:
    if position not in MODEL_PATHS:
        raise ValueError(f"Unknown position: {position}")

    model_path = Path(MODEL_PATHS[position])

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found for {position}: {model_path}")

    _log(f"Loading model for {position}: {model_path.name}", verbose)

    if model_path.suffix.lower() == ".pkl":
        return joblib.load(model_path)

    if model_path.suffix.lower() in {".keras", ".h5"}:
        return load_model(model_path)

    raise ValueError(f"Unsupported model file type for {position}: {model_path.suffix}")


def prepare_position_features(df: pd.DataFrame, position: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    feature_cols = FEATURES_BY_POSITION[position]
    missing_cols = [col for col in feature_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required features for {position}: {missing_cols}")

    return df[feature_cols].copy()


def predict_with_sklearn_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    predictions = model.predict(X)
    return np.asarray(predictions).reshape(-1)


def predict_with_keras_model(model: Any, X: pd.DataFrame) -> np.ndarray:
    X_array = X.to_numpy(dtype=float)
    X_array = X_array.reshape((X_array.shape[0], 1, X_array.shape[1]))
    predictions = model.predict(X_array, verbose=0)
    return np.asarray(predictions).reshape(-1)


def predict_for_position(position: str, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    model = load_position_model(position=position, verbose=verbose)
    X = prepare_position_features(df=df, position=position)

    if position == "FWD":
        preds = predict_with_keras_model(model=model, X=X)
        model_used = "LSTM"
    else:
        preds = predict_with_sklearn_model(model=model, X=X)
        model_used = "Linear Regression"

    output_df = df.copy()
    output_df["position"] = position
    output_df["predicted_points"] = preds
    output_df["model_used"] = model_used
    return output_df


def add_player_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add reusable player risk indicators from availability and recent involvement fields.
    Missing columns are ignored so the prediction pipeline remains defensive.
    """
    if df.empty:
        return df.copy()

    output_df = df.copy()
    high_risk = pd.Series(False, index=output_df.index)
    medium_risk = pd.Series(False, index=output_df.index)
    flag_lists = pd.Series([[] for _ in range(len(output_df))], index=output_df.index)
    news_text = pd.Series("", index=output_df.index)

    if "news" in output_df.columns:
        news_text = output_df["news"].fillna("").astype(str).str.strip()

    def add_flag(mask: pd.Series, label: str | pd.Series) -> None:
        for idx in output_df.index[mask.fillna(False)]:
            flag = label.at[idx] if isinstance(label, pd.Series) else label
            flag = str(flag).strip()
            if flag and flag.lower() not in {"none", "nan"} and flag not in flag_lists.at[idx]:
                flag_lists.at[idx].append(flag)

    def news_or_fallback(fallback: str) -> pd.Series:
        return news_text.where(news_text.ne(""), fallback)

    if "status" in output_df.columns:
        status = output_df["status"].astype("string").str.lower().str.strip()
        status_risk = status.notna() & status.ne("a")
        high_risk = high_risk | status_risk
        add_flag(status_risk, news_or_fallback("Unavailable"))

    if "chance_of_playing_next_round" in output_df.columns:
        chance = pd.to_numeric(output_df["chance_of_playing_next_round"], errors="coerce")
        chance_risk = chance.notna() & chance.lt(75)
        high_risk = high_risk | chance_risk
        no_existing_flag = flag_lists.map(len).eq(0)
        add_flag(chance_risk & no_existing_flag, news_or_fallback("Chance < 75"))

    if "minutes_rolling3" in output_df.columns:
        minutes_rolling3 = pd.to_numeric(output_df["minutes_rolling3"], errors="coerce")
        recent_minutes_risk = minutes_rolling3.notna() & minutes_rolling3.lt(15) & ~high_risk
        medium_risk = medium_risk | recent_minutes_risk
        add_flag(recent_minutes_risk, "Limited recent minutes")

    output_df["risk_level"] = np.select(
        [high_risk, medium_risk & ~high_risk],
        ["High", "Medium"],
        default="Low",
    )
    output_df["risk_flags"] = flag_lists.map(lambda flags: ", ".join(flags) if flags else "None")

    return output_df


def build_predictions_table(
    use_cache: bool = True,
    verbose: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> pd.DataFrame:
    _emit_progress(progress_callback, 0.01, "Building latest inference tables...")

    def inference_progress(local_progress: float, message: str) -> None:
        mapped_progress = 0.02 + (0.70 * local_progress)
        _emit_progress(progress_callback, mapped_progress, message)

    inference_tables = get_latest_round_inference_tables(
        use_cache=use_cache,
        verbose=verbose,
        progress_callback=inference_progress,
    )

    outputs: list[pd.DataFrame] = []
    positions = ["GK", "DEF", "MID", "FWD"]

    stage_points = {
        "GK": 0.78,
        "DEF": 0.84,
        "MID": 0.90,
        "FWD": 0.96,
    }

    for position in positions:
        _emit_progress(progress_callback, stage_points[position], f"Running predictions for {position}...")
        pred_df = predict_for_position(
            position=position,
            df=inference_tables[position],
            verbose=verbose,
        )
        if not pred_df.empty:
            outputs.append(pred_df)

    if not outputs:
        return pd.DataFrame()

    predictions_df = pd.concat(outputs, ignore_index=True)

    if "value" in predictions_df.columns and "price_m" not in predictions_df.columns:
        predictions_df["price_m"] = predictions_df["value"]

    if "price_m" in predictions_df.columns:
        predictions_df["price_m"] = pd.to_numeric(predictions_df["price_m"], errors="coerce")
        if predictions_df["price_m"].dropna().median() > 20:
            predictions_df["price_m"] = predictions_df["price_m"] / 10.0

    predictions_df["predicted_points"] = predictions_df["predicted_points"].astype(float)
    predictions_df = add_player_risk_indicators(predictions_df)
    predictions_df = predictions_df.sort_values(
        ["position", "predicted_points"],
        ascending=[True, False],
    ).reset_index(drop=True)

    predictions_df["source_round"] = predictions_df["round"]

    _emit_progress(progress_callback, 1.0, "Live predictions ready.")
    return predictions_df


def inspect_predictions(use_cache: bool = True, verbose: bool = True) -> pd.DataFrame:
    return build_predictions_table(use_cache=use_cache, verbose=verbose)
