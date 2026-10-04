from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import xarray as xr
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "west_bengal_climate_final.nc"
STATS_PATH = ROOT / "data" / "normalization_stats.csv"
MODEL_PATH = ROOT / "models" / "varuna_convlstm_attention_1day.keras"
BOUNDARY_PATH = ROOT / "data" / "state_NWIC.GeoJSON"

VARIABLES = ["rainfall", "tmax", "tmin"]

# These are approximate reference labels for coarse grid cells, not station locations.
# Keep the coordinate keys exactly aligned with the prepared 7x5 grid.
GRID_REFERENCE_NAMES = {
    (22.5, 86.5): ("Jhargram-side fringe", "Jhargram region"),
    (22.5, 87.5): ("Midnapore region", "Midnapore"),
    (22.5, 88.5): ("Kolkata-side south", "Kolkata region"),
    (23.5, 85.5): ("Western boundary fringe", "West fringe"),
    (23.5, 86.5): ("Purulia-side region", "Purulia region"),
    (23.5, 87.5): ("Bankura-side region", "Bankura region"),
    (23.5, 88.5): ("Nadia-side region", "Nadia region"),
    (24.5, 87.5): ("Murshidabad west", "Murshidabad W"),
    (24.5, 88.5): ("Murshidabad east", "Murshidabad E"),
    (25.5, 87.5): ("Malda-side west", "Malda W"),
    (25.5, 88.5): ("Malda-side east", "Malda E"),
    (26.5, 87.5): ("Northern fringe cell", "North fringe"),
    (26.5, 89.5): ("Alipurduar-side region", "Alipurduar region"),
    (27.5, 88.5): ("Kalimpong-side fringe", "Kalimpong region"),
}


@tf.keras.utils.register_keras_serializable()
class TemporalAttentionPooling(layers.Layer):
    def call(self, inputs):
        features, weights = inputs
        weights = tf.reshape(
            weights,
            [tf.shape(weights)[0], tf.shape(weights)[1], 1, 1, 1],
        )
        return tf.reduce_sum(features * weights, axis=1)


def load_resources():
    for path in [DATA_PATH, STATS_PATH, MODEL_PATH]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    ds = xr.open_dataset(DATA_PATH)
    stats_df = pd.read_csv(STATS_PATH)
    stats = stats_df.set_index("variable")

    means = np.array([stats.loc[v, "mean"] for v in VARIABLES], dtype=np.float32)
    stds = np.array([stats.loc[v, "std"] for v in VARIABLES], dtype=np.float32)

    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError("Normalization statistics are invalid.")

    mask = ds["west_bengal_mask"].values.astype(bool)
    if mask.shape != (7, 5) or int(mask.sum()) != 14:
        raise ValueError(f"Unexpected mask shape/count: {mask.shape}, {mask.sum()}")

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TemporalAttentionPooling": TemporalAttentionPooling},
        compile=False,
    )
    return ds, mask, means, stds, model, stats_df


def _get_history(ds, mask, target_date):
    dates = pd.to_datetime(ds.time.values).normalize()
    target = pd.Timestamp(target_date).normalize()
    prior = np.flatnonzero(dates < target)

    if len(prior) < 7:
        raise ValueError("The target date must have at least seven prior observations.")
    idx = prior[-7:]
    if not np.array_equal(np.diff(ds.time.values[idx].astype("datetime64[D]")), np.timedelta64(1, "D") * np.ones(6)):
        raise ValueError("The seven prior observations are not consecutive calendar days.")

    raw = np.stack(
        [ds[v].isel(time=idx).values for v in VARIABLES],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(raw[:, mask, :]).all():
        raise ValueError("Missing values found inside selected grid cells.")

    return raw, pd.to_datetime(ds.time.values[idx])


def _make_model_input(raw, mask, means, stds):
    norm = (raw - means[None, None, None, :]) / stds[None, None, None, :]
    norm[:, ~mask, :] = 0.0
    mask_channel = np.broadcast_to(
        mask.astype(np.float32)[None, :, :, None],
        (7, 7, 5, 1),
    )
    x = np.concatenate([norm, mask_channel], axis=-1)
    return x[None].astype(np.float32)


def _predict_grid(model, x, means, stds, mask):
    pred_norm = model.predict(x, verbose=0)
    if pred_norm.shape != (1, 1, 7, 5, 3):
        raise ValueError(f"Unexpected model output shape: {pred_norm.shape}")

    pred = pred_norm[0, 0] * stds[None, None, :] + means[None, None, :]
    pred[..., 0] = np.maximum(pred[..., 0], 0.0)

    # Masked cells are not part of the study domain.
    pred[~mask, :] = np.nan

    # Keep Tmax >= Tmin as a simple physical consistency safeguard.
    # This adjustment should be disclosed as post-processing.
    swap_mask = mask & (pred[:, :, 1] < pred[:, :, 2])
    if np.any(swap_mask):
        mid = (pred[:, :, 1] + pred[:, :, 2]) / 2.0
        pred[:, :, 1] = np.where(swap_mask, mid, pred[:, :, 1])
        pred[:, :, 2] = np.where(swap_mask, mid, pred[:, :, 2])
    return pred.astype(np.float32)


def forecast_for_target_date(ds, mask, means, stds, model, target_date):
    raw, history_dates = _get_history(ds, mask, target_date)
    x = _make_model_input(raw, mask, means, stds)
    pred_grid = _predict_grid(model, x, means, stds, mask)
    return pred_grid, pred_grid


def forecast_next_days(
    ds, mask, means, stds, model, as_of_date, horizon=3
):
    """Forecast the next N days after the selected last-observed date."""
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1 day.")

    dates = pd.to_datetime(ds.time.values).normalize()
    as_of = pd.Timestamp(as_of_date).normalize()

    prior = np.flatnonzero(dates <= as_of)

    if len(prior) < 7:
        raise ValueError("At least 7 days of history are required.")

    idx = prior[-7:]
    history_dates = dates[idx]

    if not np.all(np.diff(history_dates.values) == np.timedelta64(1, "D")):
        raise ValueError("The last 7 observations must be consecutive days.")

    history = np.stack(
        [ds[v].isel(time=idx).values for v in VARIABLES],
        axis=-1,
    ).astype(np.float32)

    if not np.isfinite(history[:, mask, :]).all():
        raise ValueError("Missing values in the selected grid cells.")

    forecasts = []

    for day in range(1, horizon + 1):
        x = _make_model_input(history, mask, means, stds)
        pred_grid = _predict_grid(model, x, means, stds, mask)

        forecast_date = as_of + pd.Timedelta(days=day)

        forecasts.append({
            "date": forecast_date,
            "grid": pred_grid.copy(),
            "regional_means": regional_means(pred_grid, mask),
        })

        # Roll the window forward: the next forecast uses this prediction.
        history = np.concatenate(
            [history[1:], pred_grid[np.newaxis, ...]],
            axis=0,
        )

    return forecasts

def regional_means(grid, mask):
    result = {}
    for i, var in enumerate(VARIABLES):
        vals = grid[:, :, i][mask]
        result[var] = float(np.nanmean(vals))
    return result


def build_grid_table(ds, mask, grid, one_channel=False):
    rows = []
    latitudes = ds.latitude.values
    longitudes = ds.longitude.values
    for i, lat in enumerate(latitudes):
        for j, lon in enumerate(longitudes):
            if not mask[i, j]:
                continue
            label, short = GRID_REFERENCE_NAMES.get(
                (round(float(lat), 1), round(float(lon), 1)),
                (f"Grid cell {lat:.1f}, {lon:.1f}", f"{lat:.1f}, {lon:.1f}"),
            )
            if one_channel:
                vals = {"impact": float(grid[i, j, 0])}
            else:
                vals = {
                    "rainfall": float(grid[i, j, 0]),
                    "tmax": float(grid[i, j, 1]),
                    "tmin": float(grid[i, j, 2]),
                }
            rows.append({
                "reference_name": label,
                "short_name": short,
                "latitude": float(lat),
                "longitude": float(lon),
                **vals,
            })
    return pd.DataFrame(rows)


def get_available_target_dates(ds):
    return pd.to_datetime(ds.time.values)


# def run_scenario(ds, mask, means, stds, model, target_date,
#                  rainfall_change=0.0, tmax_change=0.0, tmin_change=0.0):
#     raw, _ = _get_history(ds, mask, target_date)
#     baseline_x = _make_model_input(raw.copy(), mask, means, stds)

#     scenario_raw = raw.copy()
#     scenario_raw[-1, mask, 0] += float(rainfall_change)
#     scenario_raw[-1, mask, 1] += float(tmax_change)
#     scenario_raw[-1, mask, 2] += float(tmin_change)
#     scenario_raw[..., 0] = np.maximum(scenario_raw[..., 0], 0.0)
#     scenario_x = _make_model_input(scenario_raw, mask, means, stds)

#     baseline_grid = _predict_grid(model, baseline_x, means, stds, mask)
#     scenario_grid = _predict_grid(model, scenario_x, means, stds, mask)

#     return (
#         regional_means(baseline_grid, mask),
#         regional_means(scenario_grid, mask),
#         baseline_grid,
#         scenario_grid,
#     )



def run_scenario(
    ds,
    mask,
    means,
    stds,
    model,
    target_date,
    rainfall_change=0.0,
    tmax_change=0.0,
    tmin_change=0.0,
    location_index=None,
):
    """
    Compare baseline and what-if forecasts.

    If location_index=(row, col) is supplied, changes are applied
    only to that grid cell on the final day of the 7-day history.
    Otherwise, the changes apply to every valid grid cell.
    """
    raw, _ = _get_history(ds, mask, target_date)

    baseline_x = _make_model_input(
        raw.copy(), mask, means, stds
    )

    scenario_raw = raw.copy()

    if location_index is None:
        # Backward-compatible regional scenario.
        scenario_raw[-1, mask, 0] += float(rainfall_change)
        scenario_raw[-1, mask, 1] += float(tmax_change)
        scenario_raw[-1, mask, 2] += float(tmin_change)
    else:
        row, col = location_index

        if not (0 <= row < mask.shape[0] and
                0 <= col < mask.shape[1]):
            raise ValueError("Selected grid cell is out of bounds.")

        if not mask[row, col]:
            raise ValueError("Selected grid cell is outside the study mask.")

        # Change only the selected cell, on the final observed day.
        scenario_raw[-1, row, col, 0] += float(rainfall_change)
        scenario_raw[-1, row, col, 1] += float(tmax_change)
        scenario_raw[-1, row, col, 2] += float(tmin_change)

    # Rainfall cannot be negative.
    scenario_raw[..., 0] = np.maximum(
        scenario_raw[..., 0], 0.0
    )

    scenario_x = _make_model_input(
        scenario_raw, mask, means, stds
    )

    baseline_grid = _predict_grid(
        model, baseline_x, means, stds, mask
    )
    scenario_grid = _predict_grid(
        model, scenario_x, means, stds, mask
    )

    if location_index is None:
        baseline = regional_means(baseline_grid, mask)
        scenario = regional_means(scenario_grid, mask)
    else:
        row, col = location_index

        baseline = {
            "rainfall": float(baseline_grid[row, col, 0]),
            "tmax": float(baseline_grid[row, col, 1]),
            "tmin": float(baseline_grid[row, col, 2]),
        }
        scenario = {
            "rainfall": float(scenario_grid[row, col, 0]),
            "tmax": float(scenario_grid[row, col, 1]),
            "tmin": float(scenario_grid[row, col, 2]),
        }

    return (
        baseline,
        scenario,
        baseline_grid,
        scenario_grid,
    )

