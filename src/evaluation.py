from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
SEQUENCE_DIR = DATA_DIR / "sequences"
MODEL_DIR = ROOT / "models"


# ============================================================
# CONFIGURATION
# ============================================================

VARIABLES = [
    "rainfall",
    "tmax",
    "tmin"
]

DISPLAY_NAMES = {
    "rainfall": "Rainfall",
    "tmax": "Tmax",
    "tmin": "Tmin"
}

UNITS = {
    "rainfall": "mm/day",
    "tmax": "°C",
    "tmin": "°C"
}


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    path = SEQUENCE_DIR / "test_1day_fixed.npz"

    if not path.exists():
        raise FileNotFoundError(
            f"Test sequence file not found:\n{path}"
        )

    data = np.load(path, allow_pickle=False)

    print("Loaded:", path)
    print("Available arrays:", data.files)

    return data


# ============================================================
# INSPECT TEST DATA
# ============================================================

def inspect_test_data():

    data = load_test_data()

    information = []

    for key in data.files:

        arr = data[key]

        information.append({
            "Array": key,
            "Shape": str(arr.shape),
            "Dtype": str(arr.dtype),
            "Size": arr.size
        })

    return pd.DataFrame(information)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(model_path):

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{model_path}"
        )

    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    return model


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    valid = (
        np.isfinite(y_true) &
        np.isfinite(y_pred)
    )

    y_true = y_true[valid]
    y_pred = y_pred[valid]

    if len(y_true) == 0:

        return {
            "MAE": np.nan,
            "RMSE": np.nan
        }

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse)
    }


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_model(
    model_path,
    X_test,
    y_test
):

    model = load_model(model_path)

    print("\nEvaluating:")
    print(model_path)

    print(
        "Expected input:",
        model.input_shape
    )

    print(
        "Test input:",
        X_test.shape
    )

    predictions = model.predict(
        X_test,
        verbose=0
    )

    print(
        "Prediction shape:",
        predictions.shape
    )

    results = []

    # --------------------------------------------------------
    # Handle the common VARUNA one-day output format
    # --------------------------------------------------------

    if predictions.ndim == 5:

        predictions = predictions[:, 0]

    if y_test.ndim == 5:

        y_test = y_test[:, 0]

    # Expected:
    #
    # samples × latitude × longitude × variables
    #
    # --------------------------------------------------------

    for variable_index, variable in enumerate(VARIABLES):

        true_values = (
            y_test[..., variable_index]
        )

        predicted_values = (
            predictions[..., variable_index]
        )

        metrics = calculate_metrics(
            true_values,
            predicted_values
        )

        results.append({

            "Model": Path(model_path).stem,

            "Variable":
                DISPLAY_NAMES[variable],

            "Unit":
                UNITS[variable],

            "MAE":
                metrics["MAE"],

            "RMSE":
                metrics["RMSE"]

        })

    return pd.DataFrame(results)


# ============================================================
# EVALUATE ALL MODELS
# ============================================================

def evaluate_all_models(
    X_test,
    y_test
):

    model_files = {

        "LSTM":
            MODEL_DIR / "varuna_lstm_1day.keras",

        "ConvLSTM":
            MODEL_DIR / "varuna_convlstm_1day.keras",

        "ConvLSTM + Attention":
            MODEL_DIR /
            "varuna_convlstm_attention_1day.keras"
    }

    all_results = []

    for model_name, model_path in model_files.items():

        if not model_path.exists():

            print(
                f"Skipping missing model: {model_path}"
            )

            continue

        result = evaluate_model(
            model_path,
            X_test,
            y_test
        )

        result["Model"] = model_name

        all_results.append(result)

    if not all_results:

        raise RuntimeError(
            "No trained models were found."
        )

    return pd.concat(
        all_results,
        ignore_index=True
    )