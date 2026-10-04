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

Hazard classification:
    LOW + LOW          -> NONE
    FLOOD active only  -> FLOOD
    DROUGHT active only -> DROUGHT
    Both active        -> COMPOUND

A hazard is considered ACTIVE when its risk level is
MODERATE, HIGH, or EXTREME.

Alert levels:
    0 = LOW
    1 = MODERATE
    2 = HIGH
    3 = EXTREME
"""

from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[3]
OUTPUT_DIR = BASE_DIR / "outputs"

FLOOD_FILE = OUTPUT_DIR / "flood_risk_alerts.csv"
DROUGHT_FILE = OUTPUT_DIR / "drought_risk_alerts.csv"

COMBINED_FILE = OUTPUT_DIR / "combined_risk_alerts.csv"
LATEST_FILE = OUTPUT_DIR / "latest_risk_alerts.csv"
SUMMARY_FILE = OUTPUT_DIR / "dashboard_risk_summary.json"


# ============================================================
# RISK MAPPING
# ============================================================

RISK_TO_SCORE = {
    "LOW": 0,
    "MODERATE": 1,
    "HIGH": 2,
    "EXTREME": 3,
}

SCORE_TO_RISK = {
    0: "LOW",
    1: "MODERATE",
    2: "HIGH",
    3: "EXTREME",
}


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    if not FLOOD_FILE.exists():
        raise FileNotFoundError(
            f"Missing flood output: {FLOOD_FILE}"
        )

    if not DROUGHT_FILE.exists():
        raise FileNotFoundError(
            f"Missing drought output: {DROUGHT_FILE}"
        )

    flood = pd.read_csv(FLOOD_FILE)
    drought = pd.read_csv(DROUGHT_FILE)

    for df in (flood, drought):

        df["date"] = pd.to_datetime(
            df["date"]
        ).dt.normalize()

        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )

        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )

    return flood, drought


# ============================================================
# PREPARE FLOOD DATA
# ============================================================

def prepare_flood(flood):

    cols = [
        "date",
        "latitude",
        "longitude",
        "rainfall",
        "rainfall_3day",
        "rainfall_7day",
        "daily_score",
        "rainfall_3day_score",
        "rainfall_7day_score",
        "flood_score",
        "flood_risk_level",
    ]

    cols = [
        c for c in cols
        if c in flood.columns
    ]

    out = flood[cols].copy()

    out = out.rename(
        columns={
            "rainfall": "flood_rainfall",
            "rainfall_3day": "flood_rainfall_3day",
            "rainfall_7day": "flood_rainfall_7day",
            "daily_score": "flood_daily_score",
            "rainfall_3day_score": "flood_3day_score",
            "rainfall_7day_score": "flood_7day_score",
        }
    )

    if "flood_score" in out.columns:

        out["flood_score"] = pd.to_numeric(
            out["flood_score"],
            errors="coerce"
        )

    if "flood_risk_level" in out.columns:

        out["flood_risk_score"] = (
            out["flood_risk_level"]
            .map(RISK_TO_SCORE)
            .astype("Int64")
        )

    return out


# ============================================================
# PREPARE DROUGHT DATA
# ============================================================

def prepare_drought(drought):

    cols = [
        "date",
        "latitude",
        "longitude",
        "rainfall",
        "rainfall_30day",
        "rainfall_60day",
        "rainfall_90day",
        "dry_day",
        "dry_streak",
        "rainfall_30day_deficit_pct",
        "rainfall_60day_deficit_pct",
        "rainfall_90day_deficit_pct",
        "drought_score",
        "drought_risk_level",
    ]

    cols = [
        c for c in cols
        if c in drought.columns
    ]

    out = drought[cols].copy()

    out = out.rename(
        columns={
            "rainfall": "drought_rainfall",
        }
    )

    if "drought_score" in out.columns:

        out["drought_score"] = pd.to_numeric(
            out["drought_score"],
            errors="coerce"
        )

    if "drought_risk_level" in out.columns:

        out["drought_risk_score"] = (
            out["drought_risk_level"]
            .map(RISK_TO_SCORE)
            .astype("Int64")
        )

    return out


# ============================================================
# ALERT MESSAGE
# ============================================================

def make_alert_message(row):

    level = row["combined_risk_level"]
    hazard = row["hazard_type"]

    flood_level = row.get(
        "flood_risk_level"
    )

    drought_level = row.get(
        "drought_risk_level"
    )

    # Both flood and drought are active
    if hazard == "COMPOUND":

        return (
            f"Compound flood and drought conditions detected. "
            f"Flood: {flood_level}; "
            f"Drought: {drought_level}; "
            f"Combined level: {level}."
        )

    # Flood is the active hazard
    if hazard == "FLOOD":

        return (
            f"Flood risk level: {level}."
        )

    # Drought is the active hazard
    if hazard == "DROUGHT":

        return (
            f"Drought risk level: {level}."
        )

    # No significant hazard
    return (
        f"Risk level: {level}."
    )


# ============================================================
# BUILD COMBINED DATA
# ============================================================

def build_combined(flood, drought):

    f = prepare_flood(flood)
    d = prepare_drought(drought)

    keys = [
        "date",
        "latitude",
        "longitude",
    ]

    # Outer merge deliberately preserves observations that
    # exist in only one risk module.
    combined = pd.merge(
        f,
        d,
        on=keys,
        how="outer",
        suffixes=(
            "_flood",
            "_drought"
        ),
    )

    # --------------------------------------------------------
    # FLOOD SCORE
    # --------------------------------------------------------

    flood_score = pd.to_numeric(
        combined.get(
            "flood_risk_score"
        ),
        errors="coerce"
    )

    # --------------------------------------------------------
    # DROUGHT SCORE
    # --------------------------------------------------------

    drought_score = pd.to_numeric(
        combined.get(
            "drought_risk_score"
        ),
        errors="coerce"
    )

    # --------------------------------------------------------
    # COMBINED SCORE
    # --------------------------------------------------------

    combined["combined_score"] = pd.concat(
        [
            flood_score,
            drought_score
        ],
        axis=1
    ).max(
        axis=1,
        skipna=True
    )

    combined["combined_score"] = (
        combined["combined_score"]
        .fillna(0)
        .astype(int)
    )

    combined["combined_risk_level"] = (
        combined["combined_score"]
        .map(SCORE_TO_RISK)
    )

    # --------------------------------------------------------
    # DETERMINE WHETHER FLOOD/DROUGHT DATA EXISTS
    # --------------------------------------------------------

    flood_present = flood_score.notna()

    drought_present = drought_score.notna()

    # --------------------------------------------------------
    # ACTIVE HAZARDS
    #
    # Score >= 1 means:
    # MODERATE / HIGH / EXTREME
    # --------------------------------------------------------

    flood_active = (
        flood_present
        & (flood_score >= 1)
    )

    drought_active = (
        drought_present
        & (drought_score >= 1)
    )

    # --------------------------------------------------------
    # HAZARD TYPE
    # --------------------------------------------------------

    combined["hazard_type"] = "NONE"

    # Flood only
    combined.loc[
        flood_active & ~drought_active,
        "hazard_type"
    ] = "FLOOD"

    # Drought only
    combined.loc[
        drought_active & ~flood_active,
        "hazard_type"
    ] = "DROUGHT"

    # Both hazards active
    combined.loc[
        flood_active & drought_active,
        "hazard_type"
    ] = "COMPOUND"

    # --------------------------------------------------------
    # ALERT PRIORITY
    # --------------------------------------------------------

    combined["alert_priority"] = (
        combined["combined_risk_level"]
        .map(
            {
                "LOW": "NORMAL",
                "MODERATE": "WATCH",
                "HIGH": "WARNING",
                "EXTREME": "CRITICAL",
            }
        )
    )

    # --------------------------------------------------------
    # ALERT MESSAGE
    # --------------------------------------------------------

    combined["alert_message"] = (
        combined.apply(
            make_alert_message,
            axis=1
        )
    )

    return combined


# ============================================================
# BUILD DASHBOARD SUMMARY
# ============================================================

def make_summary(
    combined,
    flood,
    drought
):

    latest_date = (
        combined["date"].max()
    )

    latest = combined[
        combined["date"] == latest_date
    ].copy()

    # --------------------------------------------------------
    # OVERALL RISK DISTRIBUTION
    # --------------------------------------------------------

    risk_counts = (
        combined["combined_risk_level"]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MODERATE",
                "HIGH",
                "EXTREME",
            ],
            fill_value=0
        )
        .astype(int)
        .to_dict()
    )

    # --------------------------------------------------------
    # LATEST DAY RISK DISTRIBUTION
    # --------------------------------------------------------

    latest_counts = (
        latest["combined_risk_level"]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MODERATE",
                "HIGH",
                "EXTREME",
            ],
            fill_value=0
        )
        .astype(int)
        .to_dict()
    )

    # --------------------------------------------------------
    # HAZARD DISTRIBUTION
    # --------------------------------------------------------

    hazard_counts = (
        combined["hazard_type"]
        .value_counts()
        .to_dict()
    )

    # --------------------------------------------------------
    # HIGH / EXTREME ALERTS
    # --------------------------------------------------------

    critical = combined[
        combined["combined_risk_level"].isin(
            [
                "HIGH",
                "EXTREME",
            ]
        )
    ].copy()

    top_alerts = (
        critical
        .sort_values(
            [
                "combined_score",
                "date"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(10)
    )

    # --------------------------------------------------------
    # TOP ALERT LIST
    # --------------------------------------------------------

    top_alerts_list = []

    for _, r in top_alerts.iterrows():

        top_alerts_list.append(
            {
                "date": r["date"].strftime(
                    "%Y-%m-%d"
                ),

                "latitude": float(
                    r["latitude"]
                ),

                "longitude": float(
                    r["longitude"]
                ),

                "hazard_type": r[
                    "hazard_type"
                ],

                "risk_level": r[
                    "combined_risk_level"
                ],

                "flood_risk": (
                    None
                    if pd.isna(
                        r.get(
                            "flood_risk_level"
                        )
                    )
                    else str(
                        r.get(
                            "flood_risk_level"
                        )
                    )
                ),

                "drought_risk": (
                    None
                    if pd.isna(
                        r.get(
                            "drought_risk_level"
                        )
                    )
                    else str(
                        r.get(
                            "drought_risk_level"
                        )
                    )
                ),

                "alert_message": r[
                    "alert_message"
                ],
            }
        )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    return {

        "module":
            "VARUNA Combined Risk Alerts",

        "generated_at":
            pd.Timestamp.now().isoformat(),

        "dataset_period": {

            "start":
                combined["date"]
                .min()
                .strftime(
                    "%Y-%m-%d"
                ),

            "end":
                combined["date"]
                .max()
                .strftime(
                    "%Y-%m-%d"
                ),
        },

        "total_combined_observations":
            int(
                len(combined)
            ),

        "grid_cells":
            int(
                combined[
                    [
                        "latitude",
                        "longitude"
                    ]
                ]
                .drop_duplicates()
                .shape[0]
            ),

        "latest_date":
            latest_date.strftime(
                "%Y-%m-%d"
            ),

        "latest_day_observations":
            int(
                len(latest)
            ),

        "combined_risk_distribution":
            risk_counts,

        "latest_risk_distribution":
            latest_counts,

        "hazard_type_distribution":
            {
                str(k): int(v)
                for k, v in hazard_counts.items()
            },

        "flood_input_observations":
            int(
                len(flood)
            ),

        "drought_input_observations":
            int(
                len(drought)
            ),

        "high_or_extreme_observations":
            int(
                len(critical)
            ),

        "top_alerts":
            top_alerts_list,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    print(
        "========== VARUNA COMBINED RISK ALERTS =========="
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    flood, drought = load_data()

    print(
        f"Flood observations   : {len(flood)}"
    )

    print(
        f"Drought observations : {len(drought)}"
    )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    combined = build_combined(
        flood,
        drought
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = make_summary(
        combined,
        flood,
        drought
    )

    # --------------------------------------------------------
    # SAVE COMBINED CSV
    # --------------------------------------------------------

    combined.to_csv(
        COMBINED_FILE,
        index=False
    )

    # --------------------------------------------------------
    # LATEST SIGNIFICANT ALERTS
    # --------------------------------------------------------

    latest_alerts = combined[
        combined[
            "combined_risk_level"
        ].isin(
            [
                "MODERATE",
                "HIGH",
                "EXTREME",
            ]
        )
    ].copy()

    latest_alerts = (
        latest_alerts
        .sort_values(
            [
                "date",
                "combined_score"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    latest_alerts.to_csv(
        LATEST_FILE,
        index=False
    )

    # --------------------------------------------------------
    # SAVE JSON SUMMARY
    # --------------------------------------------------------

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print(
        "\n========== COMBINED RISK DISTRIBUTION =========="
    )

    print(
        combined[
            "combined_risk_level"
        ]
        .value_counts()
        .reindex(
            [
                "LOW",
                "MODERATE",
                "HIGH",
                "EXTREME",
            ],
            fill_value=0
        )
    )

    print(
        "\n========== HAZARD TYPE DISTRIBUTION =========="
    )

    print(
        combined[
            "hazard_type"
        ].value_counts()
    )

    print(
        "\n========== LATEST DATE =========="
    )

    print(
        summary[
            "latest_date"
        ]
    )

    latest_display = combined[
        combined["date"]
        == combined["date"].max()
    ][
        [
            "date",
            "latitude",
            "longitude",
            "flood_risk_level",
            "drought_risk_level",
            "combined_risk_level",
            "hazard_type",
            "alert_priority",
            "alert_message",
        ]
    ]

    print(
        latest_display.to_string(
            index=False
        )
    )

    print(
        "\n========== TOP HIGH / EXTREME ALERTS =========="
    )

    top_display = (
        combined[
            combined[
                "combined_risk_level"
            ].isin(
                [
                    "HIGH",
                    "EXTREME",
                ]
            )
        ][
            [
                "date",
                "latitude",
                "longitude",
                "flood_risk_level",
                "drought_risk_level",
                "combined_risk_level",
                "hazard_type",
            ]
        ]
        .sort_values(
            [
                "combined_risk_level",
                "date"
            ],
            ascending=[
                False,
                False
            ]
        )
        .head(20)
    )

    print(
        top_display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # OUTPUT PATHS
    # --------------------------------------------------------

    print(
        "\nSaved to:"
    )

    print(
        COMBINED_FILE
    )

    print(
        LATEST_FILE
    )

    print(
        SUMMARY_FILE
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()