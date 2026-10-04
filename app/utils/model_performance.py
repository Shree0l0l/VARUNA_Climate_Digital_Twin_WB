from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.forecasting import TemporalAttentionPooling


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "sequences"
    / "test_1day_fixed.npz"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "varuna_convlstm_attention_1day.keras"
)


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    data = np.load(TEST_FILE)

    X_lstm = data["X_lstm"].astype(np.float32)
    X_conv = data["X_conv"].astype(np.float32)
    y = data["y"].astype(np.float32)

    target_dates = data["target_dates"]

    mask = data["mask"]

    variables = data["variables"]

    return (
        X_lstm,
        X_conv,
        y,
        target_dates,
        mask,
        variables
    )


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    model = tf.keras.models.load_model(
    MODEL_FILE,
    custom_objects={
        "TemporalAttentionPooling": TemporalAttentionPooling
    },
    compile=False
    )
    print("Model name:", model.name)
    print("Model inputs:", model.inputs)
    print(
    "Input shapes:",
    [tuple(tensor.shape) for tensor in model.inputs]
)
    return model




# ============================================================
# GENERATE PREDICTIONS
# ============================================================


def generate_predictions(model, X_lstm, X_conv):
    import numpy as np

    # This model accepts one spatial climate input.
    expected_shape = tuple(model.input_shape[1:])
    actual_shape = tuple(X_conv.shape[1:])

    if actual_shape != expected_shape:
        raise ValueError(
            f"ConvLSTM input shape mismatch: model expects "
            f"{expected_shape}, but test data has {actual_shape}. "
            "Check the sequence-generation code to identify "
            "the missing input channel. Do not add a dummy channel."
        )

    predictions = model.predict(
        {"climate_input": X_conv},
        verbose=0
    )

    return np.asarray(predictions)


# ============================================================
# NORMALIZE PREDICTION SHAPE
# ============================================================

def prepare_prediction_shape(predictions):

    predictions = np.asarray(predictions)

    print("Raw prediction shape:", predictions.shape)

    return predictions


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    y_true_flat = y_true.reshape(-1)
    y_pred_flat = y_pred.reshape(-1)

    valid = (
        np.isfinite(y_true_flat)
        &
        np.isfinite(y_pred_flat)
    )

    y_true_flat = y_true_flat[valid]
    y_pred_flat = y_pred_flat[valid]

    mae = mean_absolute_error(
        y_true_flat,
        y_pred_flat
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true_flat,
            y_pred_flat
        )
    )

    r2 = r2_score(
        y_true_flat,
        y_pred_flat
    )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "valid_values": len(y_true_flat)
    }


# ============================================================
# COMPLETE EVALUATION
# ============================================================

def evaluate_model():

    (
        X_lstm,
        X_conv,
        y_true,
        target_dates,
        mask,
        variables
    ) = load_test_data()

    model = load_model()

    y_pred = generate_predictions(
        model,
        X_lstm,
        X_conv
    )

    y_pred = prepare_prediction_shape(
        y_pred
    )

    metrics = calculate_metrics(
        y_true,
        y_pred
    )

    return {
        "model": model,
        "X_lstm": X_lstm,
        "X_conv": X_conv,
        "y_true": y_true,
        "y_pred": y_pred,
        "target_dates": target_dates,
        "mask": mask,
        "variables": variables,
        "metrics": metrics
    }