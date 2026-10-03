"""
VARUNA - Combined Risk Alerts Module

Combines:
    outputs/flood_risk_alerts.csv
    outputs/drought_risk_alerts.csv

Outputs:
    outputs/combined_risk_alerts.csv
    outputs/dashboard_risk_summary.json
    outputs/latest_risk_alerts.csv

The module keeps flood and drought risks separate, then creates a
dashboard-ready combined alert level using the highest available
hazard severity.

Alert levels:
    0 = LOW
    1 = MODERATE
    2 = HIGH
    3 = EXTREME
"""

from pathlib import Path
import json
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

FLOOD_FILE = OUTPUT_DIR / "flood_risk_alerts.csv"
DROUGHT_FILE = OUTPUT_DIR / "drought_risk_alerts.csv"

COMBINED_FILE = OUTPUT_DIR / "combined_risk_alerts.csv"
LATEST_FILE = OUTPUT_DIR / "latest_risk_alerts.csv"
SUMMARY_FILE = OUTPUT_DIR / "dashboard_risk_summary.json"

RISK_TO_SCORE = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "EXTREME": 3,
}

SCORE_TO_RISK = {v: k for k, v in RISK_TO_SCORE.items()}


def load_data():
    if not FLOOD_FILE.exists():
        raise FileNotFoundError(f"Missing flood output: {FLOOD_FILE}")
    if not DROUGHT_FILE.exists():
        raise FileNotFoundError(f"Missing drought output: {DROUGHT_FILE}")

    flood = pd.read_csv(FLOOD_FILE)
    drought = pd.read_csv(DROUGHT_FILE)

    for df in (flood, drought):
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        df["latitude"] = pd.to_numeric(df["latitude"])
        df["longitude"] = pd.to_numeric(df["longitude"])

    return flood, drought


def prepare_flood(flood):
    cols = [
        "date", "latitude", "longitude",
        "rainfall", "rainfall_3day", "rainfall_7day",
        "daily_score", "rainfall_3day_score", "rainfall_7day_score",
        "flood_score", "flood_risk_level",
    ]
    cols = [c for c in cols if c in flood.columns]
    out = flood[cols].copy()

    out = out.rename(columns={
        "rainfall": "flood_rainfall",
        "rainfall_3day": "flood_rainfall_3day",
        "rainfall_7day": "flood_rainfall_7day",
        "daily_score": "flood_daily_score",
        "rainfall_3day_score": "flood_3day_score",
        "rainfall_7day_score": "flood_7day_score",
    })

    if "flood_score" in out:
        out["flood_score"] = pd.to_numeric(out["flood_score"], errors="coerce")
    if "flood_risk_level" in out:
        out["flood_risk_score"] = (
            out["flood_risk_level"].map(RISK_TO_SCORE).astype("Int64")
        )

    return out


def prepare_drought(drought):
    cols = [
        "date", "latitude", "longitude",
        "rainfall", "rainfall_30day", "rainfall_60day", "rainfall_90day",
        "dry_day", "dry_streak",
        "rainfall_30day_deficit_pct",
        "rainfall_60day_deficit_pct",
        "rainfall_90day_deficit_pct",
        "drought_score", "drought_risk_level",
    ]
    cols = [c for c in cols if c in drought.columns]
    out = drought[cols].copy()

    out = out.rename(columns={
        "rainfall": "drought_rainfall",
    })

    if "drought_score" in out:
        out["drought_score"] = pd.to_numeric(
            out["drought_score"], errors="coerce"
        )
    if "drought_risk_level" in out:
        out["drought_risk_score"] = (
            out["drought_risk_level"].map(RISK_TO_SCORE).astype("Int64")
        )

    return out


def build_combined(flood, drought):
    f = prepare_flood(flood)
    d = prepare_drought(drought)

    keys = ["date", "latitude", "longitude"]

    # Outer merge deliberately preserves observations that exist in only
    # one risk module (e.g. early dates before drought rolling windows exist).
    combined = pd.merge(
        f,
        d,
        on=keys,
        how="outer",
        suffixes=("_flood", "_drought"),
    )

    # Highest available hazard score becomes the combined hazard score.
    flood_score = pd.to_numeric(
        combined.get("flood_risk_score"), errors="coerce"
    )
    drought_score = pd.to_numeric(
        combined.get("drought_risk_score"), errors="coerce"
    )

    combined["combined_score"] = pd.concat(
        [flood_score, drought_score], axis=1
    ).max(axis=1, skipna=True)

    combined["combined_score"] = combined["combined_score"].fillna(0).astype(int)
    combined["combined_risk_level"] = combined["combined_score"].map(
        SCORE_TO_RISK
    )

    flood_present = flood_score.notna()
    drought_present = drought_score.notna()

    combined["hazard_type"] = "NONE"
    both = flood_present & drought_present
    flood_only = flood_present & ~drought_present
    drought_only = drought_present & ~flood_present

    combined.loc[flood_only, "hazard_type"] = "FLOOD"
    combined.loc[drought_only, "hazard_type"] = "DROUGHT"

    # If both systems have data, distinguish compound conditions only when
    # both hazards are at least MODERATE.
    compound = (
        both
        & (flood_score >= 1)
        & (drought_score >= 1)
    )
    combined.loc[both & ~compound, "hazard_type"] = "MULTI_HAZARD"
    combined.loc[compound, "hazard_type"] = "COMPOUND"

    combined["alert_priority"] = combined["combined_risk_level"].map({
        "LOW": "NORMAL",
        "MODERATE": "WATCH",
        "HIGH": "WARNING",
        "EXTREME": "CRITICAL",
    })

    combined["alert_message"] = combined.apply(make_alert_message, axis=1)

    # Dashboard-friendly ordering.
    combined = combined.sort_values(
        ["date", "latitude", "longitude"]
    ).reset_index(drop=True)

    return combined


def make_alert_message(row):
    level = row["combined_risk_level"]
    hazard = row["hazard_type"]

    flood_level = row.get("flood_risk_level")
    drought_level = row.get("drought_risk_level")

    if hazard == "COMPOUND":
        return (
            f"Compound flood and drought conditions detected. "
            f"Flood: {flood_level}; Drought: {drought_level}; "
            f"Combined level: {level}."
        )

    if hazard == "FLOOD":
        return f"Flood risk level: {level}."

    if hazard == "DROUGHT":
        return f"Drought risk level: {level}."

    if hazard == "MULTI_HAZARD":
        return (
            f"Both flood and drought modules have observations. "
            f"Combined level: {level}."
        )

    return f"Risk level: {level}."


def make_summary(combined, flood, drought):
    latest_date = combined["date"].max()

    latest = combined[combined["date"] == latest_date].copy()

    risk_counts = (
        combined["combined_risk_level"]
        .value_counts()
        .reindex(["LOW", "MODERATE", "HIGH", "EXTREME"], fill_value=0)
        .astype(int)
        .to_dict()
    )

    latest_counts = (
        latest["combined_risk_level"]
        .value_counts()
        .reindex(["LOW", "MODERATE", "HIGH", "EXTREME"], fill_value=0)
        .astype(int)
        .to_dict()
    )

    hazard_counts = (
        combined["hazard_type"]
        .value_counts()
        .to_dict()
    )

    critical = combined[
        combined["combined_risk_level"].isin(["HIGH", "EXTREME"])
    ].copy()

    top_alerts = (
        critical.sort_values(
            ["combined_score", "date"],
            ascending=[False, False]
        )
        .head(10)
    )

    top_alerts_list = []
    for _, r in top_alerts.iterrows():
        top_alerts_list.append({
            "date": r["date"].strftime("%Y-%m-%d"),
            "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"]),
            "hazard_type": r["hazard_type"],
            "risk_level": r["combined_risk_level"],
            "flood_risk": (
                None if pd.isna(r.get("flood_risk_level"))
                else str(r.get("flood_risk_level"))
            ),
            "drought_risk": (
                None if pd.isna(r.get("drought_risk_level"))
                else str(r.get("drought_risk_level"))
            ),
            "alert_message": r["alert_message"],
        })

    return {
        "module": "VARUNA Combined Risk Alerts",
        "generated_at": pd.Timestamp.now().isoformat(),
        "dataset_period": {
            "start": combined["date"].min().strftime("%Y-%m-%d"),
            "end": combined["date"].max().strftime("%Y-%m-%d"),
        },
        "total_combined_observations": int(len(combined)),
        "grid_cells": int(
            combined[["latitude", "longitude"]].drop_duplicates().shape[0]
        ),
        "latest_date": latest_date.strftime("%Y-%m-%d"),
        "latest_day_observations": int(len(latest)),
        "combined_risk_distribution": risk_counts,
        "latest_risk_distribution": latest_counts,
        "hazard_type_distribution": {
            str(k): int(v) for k, v in hazard_counts.items()
        },
        "flood_input_observations": int(len(flood)),
        "drought_input_observations": int(len(drought)),
        "high_or_extreme_observations": int(len(critical)),
        "top_alerts": top_alerts_list,
    }


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("========== VARUNA COMBINED RISK ALERTS ==========")

    flood, drought = load_data()

    print(f"Flood observations   : {len(flood)}")
    print(f"Drought observations : {len(drought)}")

    combined = build_combined(flood, drought)
    summary = make_summary(combined, flood, drought)

    combined.to_csv(COMBINED_FILE, index=False)

    latest_alerts = combined[
        combined["combined_risk_level"].isin(["MODERATE", "HIGH", "EXTREME"])
    ].copy()

    latest_alerts = latest_alerts.sort_values(
        ["date", "combined_score"],
        ascending=[False, False]
    )
    latest_alerts.to_csv(LATEST_FILE, index=False)

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n========== COMBINED RISK DISTRIBUTION ==========")
    print(
        combined["combined_risk_level"]
        .value_counts()
        .reindex(["LOW", "MODERATE", "HIGH", "EXTREME"], fill_value=0)
    )

    print("\n========== HAZARD TYPE DISTRIBUTION ==========")
    print(combined["hazard_type"].value_counts())

    print("\n========== LATEST DATE ==========")
    print(summary["latest_date"])
    print(
        combined[combined["date"] == combined["date"].max()][
            [
                "date", "latitude", "longitude",
                "flood_risk_level", "drought_risk_level",
                "combined_risk_level", "hazard_type",
                "alert_priority", "alert_message",
            ]
        ].to_string(index=False)
    )

    print("\n========== TOP HIGH / EXTREME ALERTS ==========")
    print(
        combined[
            combined["combined_risk_level"].isin(["HIGH", "EXTREME"])
        ][
            [
                "date", "latitude", "longitude",
                "flood_risk_level", "drought_risk_level",
                "combined_risk_level", "hazard_type",
            ]
        ]
        .sort_values(["combined_risk_level", "date"], ascending=[False, False])
        .head(20)
        .to_string(index=False)
    )

    print("\nSaved to:")
    print(COMBINED_FILE)
    print(LATEST_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()
