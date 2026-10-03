import xarray as xr
import pandas as pd
import numpy as np

DATA_FILE = "data/west_bengal_climate_final.nc"
BASELINE_FILE = "outputs/drought_baseline.csv"
OUTPUT_FILE = "outputs/drought_risk_features.csv"


# ==================================================
# 1. LOAD DATA
# ==================================================

ds = xr.open_dataset(DATA_FILE)

rainfall = ds["rainfall"].where(
    ds["west_bengal_mask"]
)

df = rainfall.to_dataframe(
    name="rainfall"
).reset_index()

df = df.dropna(
    subset=["rainfall"]
)

df["date"] = pd.to_datetime(df["time"])

df = df.sort_values(
    ["latitude", "longitude", "date"]
)


# ==================================================
# 2. CALCULATE ROLLING RAINFALL
# ==================================================

print("Calculating rolling rainfall...")

for days in [30, 60, 90]:

    column = f"rainfall_{days}day"

    df[column] = (
        df.groupby(
            ["latitude", "longitude"]
        )["rainfall"]
        .transform(
            lambda x: x.rolling(
                days,
                min_periods=days
            ).sum()
        )
    )


# ==================================================
# 3. MONTH
# ==================================================

df["month"] = df["date"].dt.month


# ==================================================
# 4. DRY-DAY INFORMATION
# ==================================================

# Preliminary dry-day indicator.
# This is a feature, NOT yet a final drought threshold.

df["dry_day"] = (
    df["rainfall"] <= 1.0
)


# ==================================================
# 5. CONSECUTIVE DRY DAYS
# ==================================================

def calculate_dry_streak(group):

    values = group["dry_day"].to_numpy()

    streaks = []

    current = 0

    for value in values:

        if value:
            current += 1
        else:
            current = 0

        streaks.append(current)

    return pd.Series(
        streaks,
        index=group.index
    )


df["dry_streak"] = (
    df.groupby(
        ["latitude", "longitude"],
        group_keys=False
    )
    .apply(
        calculate_dry_streak,
        include_groups=False
    )
)


# ==================================================
# 6. LOAD SEASONAL BASELINES
# ==================================================

baseline = pd.read_csv(
    BASELINE_FILE
)

print(
    f"Loaded baseline rows: {len(baseline)}"
)


# ==================================================
# 7. MERGE BASELINES
# ==================================================

baseline_columns = [
    "latitude",
    "longitude",
    "month",

    "rainfall_30day_p10",
    "rainfall_30day_p05",
    "rainfall_30day_p01",
    "rainfall_30day_median",

    "rainfall_60day_p10",
    "rainfall_60day_p05",
    "rainfall_60day_p01",
    "rainfall_60day_median",

    "rainfall_90day_p10",
    "rainfall_90day_p05",
    "rainfall_90day_p01",
    "rainfall_90day_median"
]

baseline = baseline[
    baseline_columns
]

df = df.merge(
    baseline,
    on=[
        "latitude",
        "longitude",
        "month"
    ],
    how="left"
)


# ==================================================
# 8. CALCULATE RAINFALL DEFICIT
# ==================================================

# Deficit is measured relative to the historical
# month/cell median.

for days in [30, 60, 90]:

    rainfall_column = (
        f"rainfall_{days}day"
    )

    median_column = (
        f"rainfall_{days}day_median"
    )

    deficit_column = (
        f"rainfall_{days}day_deficit_pct"
    )

    df[deficit_column] = np.where(
        df[median_column] > 0,

        (
            (
                df[median_column]
                - df[rainfall_column]
            )
            / df[median_column]
        ) * 100,

        0
    )

    # Negative deficit means rainfall is above
    # the historical median, so set it to zero.

    df[deficit_column] = (
        df[deficit_column]
        .clip(lower=0)
    )


# ==================================================
# 9. SELECT FINAL FEATURES
# ==================================================

feature_columns = [
    "date",
    "latitude",
    "longitude",

    "rainfall",

    "rainfall_30day",
    "rainfall_60day",
    "rainfall_90day",

    "dry_day",
    "dry_streak",

    "rainfall_30day_p10",
    "rainfall_30day_p05",
    "rainfall_30day_p01",
    "rainfall_30day_median",

    "rainfall_60day_p10",
    "rainfall_60day_p05",
    "rainfall_60day_p01",
    "rainfall_60day_median",

    "rainfall_90day_p10",
    "rainfall_90day_p05",
    "rainfall_90day_p01",
    "rainfall_90day_median",

    "rainfall_30day_deficit_pct",
    "rainfall_60day_deficit_pct",
    "rainfall_90day_deficit_pct"
]

features = df[feature_columns].copy()


# ==================================================
# 10. REMOVE INCOMPLETE ROLLING PERIODS
# ==================================================

features = features.dropna(
    subset=[
        "rainfall_30day",
        "rainfall_60day",
        "rainfall_90day"
    ]
)


# ==================================================
# 11. INFORMATION
# ==================================================

print("\n========== DROUGHT FEATURE DATASET ==========")

print(
    f"Rows: {len(features)}"
)

print(
    f"Columns: {len(features.columns)}"
)

print("\nColumns:")

for column in features.columns:
    print("-", column)


print("\nFirst 5 rows:")

print(
    features.head().to_string(
        index=False
    )
)


# ==================================================
# 12. SAVE
# ==================================================

features.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("==============================================")