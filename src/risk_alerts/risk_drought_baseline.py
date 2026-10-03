import xarray as xr
import pandas as pd
import numpy as np

DATA_FILE = "data/west_bengal_climate_final.nc"

ds = xr.open_dataset(DATA_FILE)

# --------------------------------------------------
# Select valid West Bengal cells
# --------------------------------------------------

rainfall = ds["rainfall"].where(ds["west_bengal_mask"])

df = rainfall.to_dataframe(
    name="rainfall"
).reset_index()

df = df.dropna(subset=["rainfall"])

df["date"] = pd.to_datetime(df["time"])

df = df.sort_values(
    ["latitude", "longitude", "date"]
)

df["month"] = df["date"].dt.month


# --------------------------------------------------
# Calculate rolling rainfall
# --------------------------------------------------

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


# Remove incomplete initial periods
df = df.dropna(
    subset=[
        "rainfall_30day",
        "rainfall_60day",
        "rainfall_90day"
    ]
)


# --------------------------------------------------
# Calculate cell + month drought baselines
# --------------------------------------------------

print("\n========== DROUGHT BASELINE ==========")

baseline_rows = []

groups = df.groupby(
    ["latitude", "longitude", "month"]
)

for (lat, lon, month), group in groups:

    row = {
        "latitude": lat,
        "longitude": lon,
        "month": month
    }

    for days in [30, 60, 90]:

        column = f"rainfall_{days}day"

        values = group[column].dropna()

        row[f"{column}_p10"] = values.quantile(0.10)
        row[f"{column}_p05"] = values.quantile(0.05)
        row[f"{column}_p01"] = values.quantile(0.01)

        row[f"{column}_median"] = values.median()

    baseline_rows.append(row)


baseline = pd.DataFrame(baseline_rows)


# --------------------------------------------------
# Display summary
# --------------------------------------------------

print(
    f"\nGrid cells: "
    f"{baseline[['latitude', 'longitude']].drop_duplicates().shape[0]}"
)

print(
    f"Months: "
    f"{baseline['month'].nunique()}"
)

print(
    f"Baseline rows: "
    f"{len(baseline)}"
)


print("\nFirst 20 baseline rows:")

print(
    baseline.head(20).to_string(
        index=False
    )
)


# --------------------------------------------------
# Save
# --------------------------------------------------

OUTPUT_FILE = "outputs/drought_baseline.csv"

baseline.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n====================================")