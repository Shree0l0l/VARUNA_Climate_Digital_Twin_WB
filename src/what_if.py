
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tensorflow.keras import layers


# --------------------------------------------------
# Paths and variable configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "data" / "west_bengal_climate_final.nc"
STATS_PATH = ROOT / "data" / "normalization_stats.csv"
MODEL_PATH = ROOT / "models" / "varuna_convlstm_attention_1day.keras"

VARIABLES = ["rainfall", "tmax", "tmin"]
VARIABLE_LABELS = {
    "rainfall": "Rainfall",
    "tmax": "Maximum temperature",
    "tmin": "Minimum temperature",
}


# --------------------------------------------------
# Custom layer used by your attention model
# --------------------------------------------------

@tf.keras.utils.register_keras_serializable()
class TemporalAttentionPooling(layers.Layer):
    def call(self, inputs):
        features, weights = inputs

        weights = tf.reshape(
            weights,
            [
                tf.shape(weights)[0],
                tf.shape(weights)[1],
                1, 1, 1,
            ],
        )

        return tf.reduce_sum(features * weights, axis=1)


# --------------------------------------------------
# Load the data, statistics, mask, and trained model
# --------------------------------------------------

def load_resources():
    for path in [DATA_PATH, STATS_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    ds = xr.open_dataset(DATA_PATH)

    stats_df = pd.read_csv(STATS_PATH).set_index("variable")

    means = np.array(
        [stats_df.loc[v, "mean"] for v in VARIABLES],
        dtype=np.float32,
    )
    stds = np.array(
        [stats_df.loc[v, "std"] for v in VARIABLES],
        dtype=np.float32,
    )

    if not np.isfinite(means).all():
        raise ValueError("Normalization means contain invalid values.")
    if not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError("Normalization standard deviations are invalid.")

    mask = ds["west_bengal_mask"].values.astype(bool)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={
            "TemporalAttentionPooling": TemporalAttentionPooling
        },
        compile=False,
    )

    return ds, mask, means, stds, model


# --------------------------------------------------
# Prepare seven days before a forecast target date
# --------------------------------------------------

def prepare_input(
    ds,
    mask,
    means,
    stds,
    target_date,
    rainfall_change=0.0,
    tmax_change=0.0,
    tmin_change=0.0,
):
    dates = pd.to_datetime(ds.time.values)
    target_date = pd.Timestamp(target_date)

    target_positions = np.flatnonzero(dates == target_date)

    if len(target_positions) != 1:
        raise ValueError(
            f"{target_date.date()} is not a unique date in the dataset."
        )

    target_idx = int(target_positions[0])

    if target_idx < 7:
        raise ValueError("The selected target date needs seven prior days.")

    # Use the seven days strictly before the target date.
    history = ds[VARIABLES].isel(
        time=slice(target_idx - 7, target_idx)
    )

    # Shape: (7 days, 7 latitude rows, 5 longitude columns, 3 variables)
    raw = np.stack(
        [history[v].values for v in VARIABLES],
        axis=-1,
    ).astype(np.float32)

    if raw.shape != (7, 7, 5, 3):
        raise ValueError(f"Unexpected input shape: {raw.shape}")

    # Only selected study-region cells should be used.
    if not np.isfinite(raw[:, mask, :]).all():
        raise ValueError(
            "Missing observations in selected cells. "
            "Do not replace missing weather observations with zero."
        )

    # Baseline input: original observations.
    baseline_raw = raw.copy()

    # Scenario input: change ONLY the final observed day.
    scenario_raw = raw.copy()

    scenario_raw[-1, mask, 0] += rainfall_change
    scenario_raw[-1, mask, 1] += tmax_change
    scenario_raw[-1, mask, 2] += tmin_change

    # Rainfall cannot be negative.
    scenario_raw[..., 0] = np.maximum(
        scenario_raw[..., 0], 0.0
    )

    def normalize_and_add_mask(raw_input):
        normalized = (raw_input - means) / stds

        # Excluded cells are placeholders, not observations.
        normalized[:, ~mask, :] = 0.0

        # Add the fourth channel expected by the ConvLSTM model.
        mask_channel = np.broadcast_to(
            mask.astype(np.float32)[None, :, :, None],
            (7, 7, 5, 1),
        )

        model_input = np.concatenate(
            [normalized, mask_channel],
            axis=-1,
        )

        # Add batch dimension.
        return model_input[None, ...].astype(np.float32)

    baseline_input = normalize_and_add_mask(baseline_raw)
    scenario_input = normalize_and_add_mask(scenario_raw)

    return (
        baseline_input,
        scenario_input,
        dates[target_idx - 7:target_idx],
    )


# --------------------------------------------------
# Predict and convert back to mm / degrees Celsius
# --------------------------------------------------

def predict_weather(model, model_input, means, stds, mask):
    prediction_norm = model.predict(model_input, verbose=0)

    # Expected shape: (1, 1, 7, 5, 3)
    if prediction_norm.shape != (1, 1, 7, 5, 3):
        raise ValueError(
            f"Unexpected model output shape: {prediction_norm.shape}"
        )

    prediction = (
        prediction_norm[0, 0] * stds[None, None, :]
        + means[None, None, :]
    )

    # Rainfall is physically non-negative.
    prediction[..., 0] = np.maximum(prediction[..., 0], 0.0)

    # Return one regional mean for each climate variable.
    return {
        "rainfall": float(np.mean(prediction[..., 0][mask])),
        "tmax": float(np.mean(prediction[..., 1][mask])),
        "tmin": float(np.mean(prediction[..., 2][mask])),
    }


# --------------------------------------------------
# Run baseline and scenario side by side
# --------------------------------------------------

def run_what_if(
    ds,
    mask,
    means,
    stds,
    model,
    target_date,
    rainfall_change=0.0,
    tmax_change=0.0,
    tmin_change=0.0,
):
    baseline_input, scenario_input, history_dates = prepare_input(
        ds=ds,
        mask=mask,
        means=means,
        stds=stds,
        target_date=target_date,
        rainfall_change=rainfall_change,
        tmax_change=tmax_change,
        tmin_change=tmin_change,
    )

    baseline = predict_weather(
        model, baseline_input, means, stds, mask
    )
    scenario = predict_weather(
        model, scenario_input, means, stds, mask
    )

    comparison = pd.DataFrame({
        "Variable": ["Rainfall (mm/day)", "Tmax (°C)", "Tmin (°C)"],
        "Baseline": [
            baseline["rainfall"],
            baseline["tmax"],
            baseline["tmin"],
        ],
        "Scenario": [
            scenario["rainfall"],
            scenario["tmax"],
            scenario["tmin"],
        ],
    })

    comparison["Difference"] = (
        comparison["Scenario"] - comparison["Baseline"]
    )

    return comparison, baseline, scenario, history_dates