import pandas as pd

# ============================================================
# DROUGHT RISK ALERT ENGINE
# ============================================================

INPUT_FILE = "outputs/drought_risk_scored.csv"
OUTPUT_FILE = "outputs/drought_risk_alerts.csv"

print("========== DROUGHT RISK ALERT ENGINE ==========")

# ------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["date"] = pd.to_datetime(df["date"])

df = df.sort_values(
    ["latitude", "longitude", "date"]
).reset_index(drop=True)

print(f"Total observations: {len(df)}")
print(
    f"Grid cells: {df[['latitude', 'longitude']].drop_duplicates().shape[0]}"
)


# ------------------------------------------------------------
# 2. IDENTIFY DROUGHT DAYS
# ------------------------------------------------------------

# Any non-LOW condition is considered a drought-alert day.

df["drought_alert"] = df["drought_risk_level"] != "LOW"


# ------------------------------------------------------------
# 3. IDENTIFY CONTINUOUS DROUGHT EVENTS
# ------------------------------------------------------------

# A new event begins when:
# - drought condition changes from False → True
# - OR there is a gap in dates
#
# This is calculated separately for every grid cell.

df["previous_date"] = df.groupby(
    ["latitude", "longitude"]
)["date"].shift(1)

df["previous_alert"] = df.groupby(
    ["latitude", "longitude"]
)["drought_alert"].shift(1)

new_event = (
    (df["drought_alert"] == True)
    &
    (
        (df["previous_alert"] != True)
        |
        ((df["date"] - df["previous_date"]).dt.days > 1)
    )
)

df["new_event"] = new_event


# ------------------------------------------------------------
# 4. CREATE EVENT IDs
# ------------------------------------------------------------

df["event_number"] = df.groupby(
    ["latitude", "longitude"]
)["new_event"].cumsum()

# Give non-alert observations no event ID.
df["drought_event_id"] = None

alert_mask = df["drought_alert"]

df.loc[alert_mask, "drought_event_id"] = (
    df.loc[alert_mask, "latitude"].astype(str)
    + "_"
    + df.loc[alert_mask, "longitude"].astype(str)
    + "_"
    + df.loc[alert_mask, "event_number"].astype(int).astype(str)
)


# ------------------------------------------------------------
# 5. CALCULATE EVENT DURATION
# ------------------------------------------------------------

event_counts = (
    df[alert_mask]
    .groupby("drought_event_id")
    .size()
    .rename("drought_event_duration")
)

df = df.join(
    event_counts,
    on="drought_event_id"
)

df["drought_event_duration"] = (
    df["drought_event_duration"].fillna(0).astype(int)
)


# ------------------------------------------------------------
# 6. SAVE ALERT DATASET
# ------------------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# 7. SUMMARY
# ------------------------------------------------------------

print("\n========== DROUGHT RISK DISTRIBUTION ==========")

print(
    df["drought_risk_level"]
    .value_counts()
    .reindex(
        ["LOW", "MODERATE", "HIGH", "EXTREME"]
    )
    .fillna(0)
)


# ------------------------------------------------------------
# 8. DROUGHT EVENT SUMMARY
# ------------------------------------------------------------

events = (
    df[df["drought_alert"]]
    .groupby("drought_event_id")
    .agg(
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        start_date=("date", "min"),
        end_date=("date", "max"),
        duration_days=("date", "count"),
        maximum_score=("drought_score", "max"),
        maximum_risk=("drought_risk_level", "max"),
        maximum_30day_deficit=(
            "rainfall_30day_deficit_pct",
            "max"
        ),
        maximum_60day_deficit=(
            "rainfall_60day_deficit_pct",
            "max"
        ),
        maximum_90day_deficit=(
            "rainfall_90day_deficit_pct",
            "max"
        ),
        maximum_dry_streak=("dry_streak", "max")
    )
    .reset_index()
)

print("\n========== DROUGHT EVENTS ==========")

print(f"Total drought events: {len(events)}")

print("\nLongest drought events:")

print(
    events
    .sort_values(
        "duration_days",
        ascending=False
    )
    .head(20)
    .to_string(index=False)
)


# ------------------------------------------------------------
# 9. EXTREME DROUGHT EVENTS
# ------------------------------------------------------------

extreme_events = events[
    events["maximum_score"] == 3
].sort_values(
    "duration_days",
    ascending=False
)

print("\n========== EXTREME DROUGHT EVENTS ==========")

print(
    extreme_events
    .head(20)
    .to_string(index=False)
)


print("\nSaved to:")
print(OUTPUT_FILE)

print("\n==============================================")