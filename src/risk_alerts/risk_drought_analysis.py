import xarray as xr
import numpy as np
import pandas as pd

DATA_FILE = "data/west_bengal_climate_final.nc"

ds = xr.open_dataset(DATA_FILE)

# --------------------------------------------------
# Select valid West Bengal grid cells
# --------------------------------------------------

mask = ds["west_bengal_mask"]

rainfall = ds["rainfall"].where(mask)

# Convert to DataFrame
df = rainfall.to_dataframe(
    name="rainfall"
).reset_index()

df = df.dropna(subset=["rainfall"])

df["date"] = pd.to_datetime(df["time"])

print("========== DROUGHT RAINFALL ANALYSIS ==========")

print("\nDataset period:")
print(df["date"].min().date(), "to", df["date"].max().date())

print("\nValid observations:", len(df))

print("Grid cells:",
      df[["latitude", "longitude"]]
      .drop_duplicates()
      .shape[0])


# --------------------------------------------------
# 1. Basic rainfall statistics
# --------------------------------------------------

print("\n1. DAILY RAINFALL STATISTICS")

print(f"Minimum : {df['rainfall'].min():.2f} mm")
print(f"Maximum : {df['rainfall'].max():.2f} mm")
print(f"Mean    : {df['rainfall'].mean():.2f} mm")
print(f"Median  : {df['rainfall'].median():.2f} mm")


# --------------------------------------------------
# 2. Dry-day analysis
# --------------------------------------------------

print("\n2. DRY-DAY ANALYSIS")

for threshold in [0, 0.1, 1, 2, 5]:

    dry_days = (df["rainfall"] <= threshold).sum()

    percentage = (
        dry_days / len(df) * 100
    )

    print(
        f"Rainfall <= {threshold:4.1f} mm : "
        f"{dry_days:6d} observations "
        f"({percentage:.2f}%)"
    )


# --------------------------------------------------
# 3. Monthly rainfall
# --------------------------------------------------

print("\n3. MONTHLY RAINFALL")

df["month"] = df["date"].dt.month

monthly = (
    df.groupby("month")["rainfall"]
    .agg(["mean", "median", "sum"])
)

print(monthly.to_string())


# --------------------------------------------------
# 4. Yearly rainfall
# --------------------------------------------------

print("\n4. YEARLY RAINFALL")

df["year"] = df["date"].dt.year

yearly = (
    df.groupby("year")["rainfall"]
    .agg(["mean", "median", "sum"])
)

print(yearly.to_string())


# --------------------------------------------------
# 5. Rolling rainfall accumulation
# --------------------------------------------------

print("\n5. ROLLING RAINFALL ACCUMULATION")

# Calculate separately for each grid cell
df = df.sort_values(
    ["latitude", "longitude", "date"]
)

df["rainfall_7day"] = (
    df.groupby(["latitude", "longitude"])["rainfall"]
    .transform(
        lambda x: x.rolling(7, min_periods=7).sum()
    )
)

df["rainfall_30day"] = (
    df.groupby(["latitude", "longitude"])["rainfall"]
    .transform(
        lambda x: x.rolling(30, min_periods=30).sum()
    )
)

df["rainfall_60day"] = (
    df.groupby(["latitude", "longitude"])["rainfall"]
    .transform(
        lambda x: x.rolling(60, min_periods=60).sum()
    )
)

df["rainfall_90day"] = (
    df.groupby(["latitude", "longitude"])["rainfall"]
    .transform(
        lambda x: x.rolling(90, min_periods=90).sum()
    )
)


for column in [
    "rainfall_7day",
    "rainfall_30day",
    "rainfall_60day",
    "rainfall_90day"
]:

    values = df[column].dropna()

    print(f"\n{column}")

    print(
        f"Minimum : {values.min():.2f} mm"
    )

    print(
        f"Mean    : {values.mean():.2f} mm"
    )

    print(
        f"Median  : {values.median():.2f} mm"
    )

    print(
        f"10th    : {values.quantile(0.10):.2f} mm"
    )

    print(
        f"5th     : {values.quantile(0.05):.2f} mm"
    )

    print(
        f"1st     : {values.quantile(0.01):.2f} mm"
    )


# --------------------------------------------------
# 6. Consecutive dry periods
# --------------------------------------------------

print("\n6. CONSECUTIVE DRY PERIOD ANALYSIS")

# Use <= 1 mm as a preliminary dry-day definition.
# This is ONLY exploratory and will NOT become our
# final drought threshold automatically.

df["dry"] = df["rainfall"] <= 1.0

def longest_dry_period(series):

    values = series.to_numpy()

    max_run = 0
    current_run = 0

    for value in values:

        if value:
            current_run += 1
            max_run = max(max_run, current_run)

        else:
            current_run = 0

    return max_run


dry_summary = (
    df.groupby(["latitude", "longitude"])["dry"]
    .apply(longest_dry_period)
)

print("\nLongest dry period by grid cell:")

print(dry_summary.to_string())


print("\nOverall maximum consecutive dry days:",
      dry_summary.max())


# --------------------------------------------------
# 7. Low rainfall percentile thresholds
# --------------------------------------------------

print("\n7. DAILY LOW-RAINFALL PERCENTILES")

print(
    f"25th percentile : "
    f"{df['rainfall'].quantile(0.25):.2f} mm"
)

print(
    f"10th percentile : "
    f"{df['rainfall'].quantile(0.10):.2f} mm"
)

print(
    f"5th percentile  : "
    f"{df['rainfall'].quantile(0.05):.2f} mm"
)

print(
    f"1st percentile  : "
    f"{df['rainfall'].quantile(0.01):.2f} mm"
)


print("\n==============================================")