import pandas as pd
import numpy as np

# ============================================================
# DROUGHT RISK SCORING
# ============================================================

INPUT_FILE = "outputs/drought_risk_features.csv"
OUTPUT_FILE = "outputs/drought_risk_scored.csv"


# ------------------------------------------------------------
# 1. LOAD DROUGHT FEATURES
# ------------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("========== DROUGHT RISK SCORING ==========")
print(f"Input observations: {len(df)}")


# ------------------------------------------------------------
# 2. DEFICIT-BASED SCORING
#
# Larger rainfall deficit = greater drought risk.
#
# Score:
# 0 = Low / normal
# 1 = Mild deficit
# 2 = Significant deficit
# 3 = Severe deficit
# ------------------------------------------------------------

def deficit_score(deficit):
    if pd.isna(deficit):
        return 0

    if deficit < 20:
        return 0
    elif deficit < 40:
        return 1
    elif deficit < 60:
        return 2
    else:
        return 3


df["rainfall_30day_score"] = df["rainfall_30day_deficit_pct"].apply(
    deficit_score
)

df["rainfall_60day_score"] = df["rainfall_60day_deficit_pct"].apply(
    deficit_score
)

df["rainfall_90day_score"] = df["rainfall_90day_deficit_pct"].apply(
    deficit_score
)


# ------------------------------------------------------------
# 3. DRY-STREAK SCORE
#
# Consecutive dry days:
#
# < 7 days   -> 0
# 7-14 days  -> 1
# 15-29 days -> 2
# >= 30 days -> 3
# ------------------------------------------------------------

def dry_streak_score(days):
    if pd.isna(days):
        return 0

    if days < 7:
        return 0
    elif days < 15:
        return 1
    elif days < 30:
        return 2
    else:
        return 3


df["dry_streak_score"] = df["dry_streak"].apply(
    dry_streak_score
)


# ------------------------------------------------------------
# 4. COMBINE DROUGHT INDICATORS
#
# We use the strongest evidence from the rainfall deficit
# periods and combine it with the dry-streak condition.
# ------------------------------------------------------------

df["rainfall_deficit_score"] = df[
    [
        "rainfall_30day_score",
        "rainfall_60day_score",
        "rainfall_90day_score"
    ]
].max(axis=1)


# Combined drought score
#
# rainfall deficit = main signal
# dry streak       = supporting signal
#
# Maximum possible score = 3

df["drought_score"] = df[
    [
        "rainfall_deficit_score",
        "dry_streak_score"
    ]
].max(axis=1)


# ------------------------------------------------------------
# 5. DROUGHT RISK LEVEL
# ------------------------------------------------------------

def drought_risk_level(score):
    if score == 0:
        return "LOW"
    elif score == 1:
        return "MODERATE"
    elif score == 2:
        return "HIGH"
    else:
        return "EXTREME"


df["drought_risk_level"] = df["drought_score"].apply(
    drought_risk_level
)


# ------------------------------------------------------------
# 6. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n========== DROUGHT SCORE DISTRIBUTION ==========")

print("\n30-day deficit score:")
print(df["rainfall_30day_score"].value_counts().sort_index())

print("\n60-day deficit score:")
print(df["rainfall_60day_score"].value_counts().sort_index())

print("\n90-day deficit score:")
print(df["rainfall_90day_score"].value_counts().sort_index())

print("\nDry streak score:")
print(df["dry_streak_score"].value_counts().sort_index())


print("\n========== DROUGHT RISK DISTRIBUTION ==========")

print(
    df["drought_risk_level"]
    .value_counts()
    .reindex(["LOW", "MODERATE", "HIGH", "EXTREME"])
    .fillna(0)
)


print("\n========== DROUGHT SCORE STATISTICS ==========")

print(df["drought_score"].describe())


# ------------------------------------------------------------
# 7. SHOW HIGHEST-RISK EVENTS
# ------------------------------------------------------------

print("\n========== TOP DROUGHT EVENTS ==========")

top_events = df.sort_values(
    by=[
        "drought_score",
        "dry_streak",
        "rainfall_90day_deficit_pct"
    ],
    ascending=False
).head(20)

print(
    top_events[
        [
            "date",
            "latitude",
            "longitude",
            "rainfall",
            "rainfall_30day",
            "rainfall_60day",
            "rainfall_90day",
            "dry_streak",
            "rainfall_30day_deficit_pct",
            "rainfall_60day_deficit_pct",
            "rainfall_90day_deficit_pct",
            "drought_score",
            "drought_risk_level"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 8. SAVE
# ------------------------------------------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("\nSaved to:")
print(OUTPUT_FILE)

print("\n==============================================")