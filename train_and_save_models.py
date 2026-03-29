from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.constants import (
    DEF_MODEL_PATH,
    FWD_MODEL_PATH,
    GK_MODEL_PATH,
    MID_MODEL_PATH,
)
from src.features import split_position_datasets, build_feature_table


def _log(message: str) -> None:
    print(message)


def _ensure_model_dirs() -> None:
    for path in [GK_MODEL_PATH, DEF_MODEL_PATH, MID_MODEL_PATH, FWD_MODEL_PATH]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)


def get_training_tables(use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """
    Build the full feature table, then split into the four training datasets.
    """
    _log("[1/6] Building full feature table for training...")
    feature_df = build_feature_table(use_cache=use_cache, verbose=True)

    if feature_df.empty:
        raise ValueError("Feature table is empty. Cannot train models.")

    _log("[2/6] Splitting training tables by position...")
    split_tables = split_position_datasets(feature_df, verbose=True)

    for position, df in split_tables.items():
        _log(f"{position} training rows: {len(df)}")

    return split_tables


def train_linear_pipeline(
    df: pd.DataFrame,
    position_name: str,
    fit_intercept: bool,
    positive: bool,
) -> Pipeline:
    """
    Train a scaled Linear Regression pipeline.
    """
    _log(f"\nTraining Linear Regression pipeline for {position_name}...")

    X = df.drop(columns=["name", "team", "round", "target"]).copy()
    y = df["target"].copy()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression(
            fit_intercept=fit_intercept,
            positive=positive
        )),
    ])

    model.fit(X, y)
    _log(f"{position_name} linear model training complete.")
    return model


def build_fwd_lstm_model(
    num_features: int,
    units: int = 16,
) -> tf.keras.Model:
    """
    Build the final FWD LSTM model with normalization embedded inside the model.
    This avoids needing a separate scaler file at inference time.
    """
    inputs = tf.keras.Input(shape=(1, num_features), name="fwd_input")

    # Normalize across the feature axis
    norm = tf.keras.layers.Normalization(axis=-1, name="feature_normalization")
    x = norm(inputs)
    x = tf.keras.layers.LSTM(units, name="lstm")(x)
    outputs = tf.keras.layers.Dense(1, name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="fwd_lstm_model")
    model.compile(optimizer="adam", loss="mse")

    return model


def train_fwd_lstm(
    df: pd.DataFrame,
    units: int = 16,
    epochs: int = 10,
    batch_size: int = 16,
) -> tf.keras.Model:
    """
    Train the final FWD LSTM model.
    """
    _log("\nTraining final FWD LSTM model...")

    X = df.drop(columns=["name", "team", "round", "target"]).copy()
    y = df["target"].copy()

    X_array = X.to_numpy(dtype=np.float32)
    y_array = y.to_numpy(dtype=np.float32)

    # reshape to [samples, timesteps=1, features]
    X_array = X_array.reshape((X_array.shape[0], 1, X_array.shape[1]))

    model = build_fwd_lstm_model(
        num_features=X_array.shape[2],
        units=units,
    )

    # Adapt normalization layer on training data
    norm_layer = model.get_layer("feature_normalization")
    norm_layer.adapt(X_array)

    model.fit(
        X_array,
        y_array,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    _log("FWD LSTM training complete.")
    return model


def save_all_models(use_cache: bool = True) -> None:
    """
    Train and save all final models.
    """
    _ensure_model_dirs()
    split_tables = get_training_tables(use_cache=use_cache)

    gk_df = split_tables["GK"]
    def_df = split_tables["DEF"]
    mid_df = split_tables["MID"]
    fwd_df = split_tables["FWD"]

    # Final tuned settings from your selected best models
    # GK: Linear Regression (fit_intercept=True, positive=True)
    # DEF: Linear Regression (fit_intercept=True, positive=True)
    # MID: Linear Regression (fit_intercept=True, positive=False)
    # FWD: LSTM (units=16, epochs=10, batch_size=16)

    _log("\n[3/6] Training GK model...")
    gk_model = train_linear_pipeline(
        df=gk_df,
        position_name="GK",
        fit_intercept=True,
        positive=True,
    )

    _log("\n[4/6] Training DEF model...")
    def_model = train_linear_pipeline(
        df=def_df,
        position_name="DEF",
        fit_intercept=True,
        positive=True,
    )

    _log("\n[5/6] Training MID model...")
    mid_model = train_linear_pipeline(
        df=mid_df,
        position_name="MID",
        fit_intercept=True,
        positive=False,
    )

    _log("\n[6/6] Training FWD model...")
    fwd_model = train_fwd_lstm(
        df=fwd_df,
        units=16,
        epochs=10,
        batch_size=16,
    )

    _log("\nSaving models to disk...")
    import joblib
    joblib.dump(gk_model, GK_MODEL_PATH)
    joblib.dump(def_model, DEF_MODEL_PATH)
    joblib.dump(mid_model, MID_MODEL_PATH)
    fwd_model.save(FWD_MODEL_PATH)

    _log(f"Saved: {GK_MODEL_PATH}")
    _log(f"Saved: {DEF_MODEL_PATH}")
    _log(f"Saved: {MID_MODEL_PATH}")
    _log(f"Saved: {FWD_MODEL_PATH}")
    _log("\nAll model files created successfully.")


if __name__ == "__main__":
    save_all_models(use_cache=True)