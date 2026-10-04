
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting import (
    load_resources,
    forecast_for_target_date,
    forecast_next_days,
    regional_means,
    build_grid_table,
    get_available_target_dates,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="VARUNA | Climate Digital Twin of West Bengal",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# VISUAL THEME
# ============================================================

st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.3rem;
            padding-bottom: 2rem;
        }

        .hero {
            padding: 1.25rem 1.5rem;
            border-radius: 16px;
            background: linear-gradient(
                120deg,
                #0b1f33 0%,
                #123f52 55%,
                #176b70 100%
            );
            color: white;
            margin-bottom: 1rem;
        }

        .hero h1 {
            margin: 0;
            font-size: 2rem;
        }

        .hero p {
            margin: .35rem 0 0 0;
            color: #d7edf0;
        }

        div[data-testid="stMetric"] {
            background: rgba(120, 150, 160, 0.08);
            border: 1px solid rgba(120, 150, 160, 0.25);
            padding: 12px 14px;
            border-radius: 12px;
        }

        .small-note {
            font-size: .85rem;
            color: #718096;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>🌦️ VARUNA</h1>
        <p>
            Climate Digital Twin · West Bengal ·
            IMD daily rainfall, Tmax and Tmin
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD DATA AND MODEL
# ============================================================

@st.cache_resource(show_spinner="Loading VARUNA dataset and trained model...")
def resources():
    return load_resources()


try:
    ds, mask, means, stds, model, stats_df = resources()

except Exception as exc:
    st.error("VARUNA could not load its data/model files.")
    st.code(str(exc))

    st.markdown(
        """
        Check that these files exist in your project:

        - `data/west_bengal_climate_final.nc`
        - `data/normalization_stats.csv`
        - `models/varuna_convlstm_attention_1day.keras`

        If your model has a different filename, update `MODEL_PATH`
        in `src/forecasting.py`.
        """
    )
    st.stop()


time_values = pd.to_datetime(ds.time.values)

first_date = time_values[0].date()
last_observed_date = time_values[-1].date()
next_target_date = (
    time_values[-1] + pd.Timedelta(days=1)
).date()


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("VARUNA controls")

page = st.sidebar.radio(
    "Navigate",
    [
        "Climate Overview",
        "Forecast & Spatial Map",
        "What-if Simulator",
        "Model Performance",
        "About VARUNA",
    ],
)

st.sidebar.caption(
    "Data source: the prepared IMD NetCDF file in this project."
)

st.sidebar.caption(
    f"Historical archive: {first_date} — {last_observed_date}"
)

st.sidebar.caption(
    f"Selected grid cells: {int(mask.sum())} / {mask.size}"
)


# This date is retained for the existing What-if Simulator.
# The three-day forecast page has its own date selector.
st.sidebar.markdown("---")
st.sidebar.caption("Date used by the What-if Simulator")

selected_target = st.sidebar.date_input(
    "Simulation target date",
    value=next_target_date,
    min_value=(time_values[7]).date(),
    max_value=next_target_date,
    key="simulation_target_date",
    help=(
        "The simulator uses the seven days immediately before "
        "this target date."
    ),
)


# ============================================================
# HELPER: GET OBSERVED GRID FOR A DATE
# ============================================================

def get_actual_grid(date):
    date64 = np.datetime64(
        pd.Timestamp(date).to_datetime64()
    )

    hits = np.flatnonzero(
        ds.time.values.astype("datetime64[ns]")
        == date64.astype("datetime64[ns]")
    )

    if len(hits) != 1:
        return None

    idx = int(hits[0])

    return np.stack(
        [
            ds[v].isel(time=idx).values
            for v in ["rainfall", "tmax", "tmin"]
        ],
        axis=-1,
    )


# ============================================================
# HELPER: ADD MAP LAYERS
# ============================================================

def add_map_layers(
    map_obj,
    grid_df,
    value_col,
    variable,
    palette="YlOrRd",
    show_labels=True,
):
    # Add the West Bengal boundary, if available.
    boundary_path = ROOT / "data" / "state_NWIC.GeoJSON"

    if boundary_path.exists():
        try:
            with open(
                boundary_path,
                "r",
                encoding="utf-8",
            ) as f:
                boundary = json.load(f)

            features = boundary.get("features", [])

            wb_features = [
                feat
                for feat in features
                if (
                    str(
                        feat.get("properties", {}).get(
                            "state_name", ""
                        )
                    ).lower()
                    == "west bengal"
                    or str(
                        feat.get("properties", {}).get(
                            "state", ""
                        )
                    ).lower()
                    in {"west bengal", "west bengal "}
                )
            ]

            geo = (
                {
                    "type": "FeatureCollection",
                    "features": wb_features,
                }
                if wb_features
                else boundary
            )

            folium.GeoJson(
                geo,
                name="West Bengal boundary",
                style_function=lambda _: {
                    "color": "#e5f3f5",
                    "weight": 2,
                    "fillColor": "#0b2538",
                    "fillOpacity": 0.08,
                },
            ).add_to(map_obj)

        except Exception as exc:
            st.warning(
                f"Could not load the state boundary: {exc}"
            )

    if grid_df.empty:
        return

    vals = grid_df[value_col].astype(float).to_numpy()
    finite = vals[np.isfinite(vals)]

    if len(finite) == 0:
        return

    lo = float(finite.min())
    hi = float(finite.max())

    if hi == lo:
        hi = lo + 1e-6

    from branca.colormap import LinearColormap

    cmap = LinearColormap(
        colors=[
            "#2c7bb6",
            "#abd9e9",
            "#ffffbf",
            "#fdae61",
            "#d7191c",
        ],
        vmin=lo,
        vmax=hi,
        caption=(
            f"{variable} "
            f"({'mm/day' if variable == 'Rainfall' else '°C'})"
        ),
    )

    cmap.add_to(map_obj)

    for _, r in grid_df.iterrows():
        value = float(r[value_col])

        unit = (
            "mm/day"
            if variable == "Rainfall"
            else "°C"
        )

        popup = (
            f"<b>{r['reference_name']}</b><br>"
            f"Grid centre: "
            f"{r['latitude']:.1f}°N, "
            f"{r['longitude']:.1f}°E<br>"
            f"{variable}: {value:.2f} {unit}"
        )

        folium.CircleMarker(
            location=[
                r["latitude"],
                r["longitude"],
            ],
            radius=9,
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color=cmap(value),
            fill_opacity=0.92,
            tooltip=(
                f"{r['reference_name']} · {value:.2f}"
            ),
            popup=folium.Popup(
                popup,
                max_width=300,
            ),
        ).add_to(map_obj)

        if show_labels:
            folium.Marker(
                location=[
                    r["latitude"],
                    r["longitude"],
                ],
                icon=folium.DivIcon(
                    html=f"""
                    <div style="
                        font-size:10px;
                        font-weight:700;
                        color:#102a43;
                        text-shadow:
                            1px 1px 2px white,
                            -1px -1px 2px white;
                        white-space:nowrap;
                        transform:translate(10px,-8px);
                    ">
                        {r['short_name']}
                    </div>
                    """
                ),
            ).add_to(map_obj)

    folium.LayerControl(collapsed=True).add_to(map_obj)


# ============================================================
# HELPER: DISPLAY A ONE-DAY FORECAST MAP
# Retained for compatibility with the existing project.
# ============================================================

def show_forecast_map(
    target_date,
    variable,
    value_kind="forecast",
):
    _, pred_grid = forecast_for_target_date(
        ds,
        mask,
        means,
        stds,
        model,
        target_date,
    )

    grid_df = build_grid_table(
        ds,
        mask,
        pred_grid,
    )

    col = {
        "Rainfall": "rainfall",
        "Tmax": "tmax",
        "Tmin": "tmin",
    }[variable]

    map_obj = folium.Map(
        location=[24.1, 88.0],
        zoom_start=7,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    add_map_layers(
        map_obj,
        grid_df,
        col,
        variable,
    )

    st_folium(
        map_obj,
        use_container_width=True,
        height=570,
        returned_objects=[],
    )

    return grid_df, pred_grid


# ============================================================
# PAGE 1: CLIMATE OVERVIEW
# ============================================================

if page == "Climate Overview":

    st.subheader("Climate overview")

    obs_date = st.date_input(
        "Observed date",
        value=last_observed_date,
        min_value=first_date,
        max_value=last_observed_date,
        key="overview_date",
    )

    actual = get_actual_grid(obs_date)

    if actual is None:
        st.warning(
            "No observed data exists for that date in the dataset."
        )
        st.stop()

    regional = regional_means(actual, mask)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Regional rainfall",
        f"{regional['rainfall']:.2f} mm/day",
    )

    c2.metric(
        "Regional Tmax",
        f"{regional['tmax']:.2f} °C",
    )

    c3.metric(
        "Regional Tmin",
        f"{regional['tmin']:.2f} °C",
    )

    c4.metric(
        "Valid grid cells",
        f"{int(mask.sum())} / {mask.size}",
    )

    st.caption(
        "Regional values are unweighted means over "
        "the 14 selected grid cells."
    )

    variable = st.selectbox(
        "Map layer",
        ["Rainfall", "Tmax", "Tmin"],
        key="overview_layer",
    )

    grid_df = build_grid_table(
        ds,
        mask,
        actual,
    )

    col = {
        "Rainfall": "rainfall",
        "Tmax": "tmax",
        "Tmin": "tmin",
    }[variable]

    map_obj = folium.Map(
        location=[24.1, 88.0],
        zoom_start=7,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    add_map_layers(
        map_obj,
        grid_df,
        col,
        variable,
    )

    st_folium(
        map_obj,
        use_container_width=True,
        height=580,
        returned_objects=[],
    )

    # Recent 30-day observed history.
    start = (
        pd.Timestamp(obs_date)
        - pd.Timedelta(days=29)
    )

    hist = ds.sel(
        time=slice(
            start,
            pd.Timestamp(obs_date),
        )
    )

    if hist.sizes.get("time", 0) > 0:
        rows = []

        for var in ["rainfall", "tmax", "tmin"]:
            arr = hist[var].values

            for i, d in enumerate(
                pd.to_datetime(hist.time.values)
            ):
                vals = arr[i][mask]

                rows.append(
                    {
                        "Date": d,
                        "Variable": var,
                        "Value": float(np.nanmean(vals)),
                    }
                )

        hist_df = pd.DataFrame(rows)

        hist_df["Variable"] = hist_df[
            "Variable"
        ].map(
            {
                "rainfall": "Rainfall",
                "tmax": "Tmax",
                "tmin": "Tmin",
            }
        )

        st.subheader("Recent observed conditions")

        fig = px.line(
            hist_df,
            x="Date",
            y="Value",
            color="Variable",
            markers=False,
        )

        fig.update_layout(
            height=350,
            legend_title_text="",
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )


# ============================================================
# PAGE 2: THREE-DAY FORECAST AND SPATIAL MAP
# ============================================================

elif page == "Forecast & Spatial Map":

    st.subheader("Three-day climate forecast")

    st.write(
        "Choose the last observed date. VARUNA will forecast "
        "the following three calendar days."
    )

    as_of_date = st.date_input(
        "Select last observed date",
        value=last_observed_date,
        min_value=(time_values[7]).date(),
        max_value=last_observed_date,
        key="forecast_as_of_date",
        help=(
            "The model uses the seven observations up to and "
            "including this date to forecast the next day. "
            "Later forecast days use earlier predictions."
        ),
    )

    st.caption(
        "Forecast period: "
        f"{(pd.Timestamp(as_of_date) + pd.Timedelta(days=1)):%d %b %Y}"
        " to "
        f"{(pd.Timestamp(as_of_date) + pd.Timedelta(days=3)):%d %b %Y}"
    )

    # Generate the next three days recursively.
    try:
        with st.spinner("Generating the three-day forecast..."):
            forecasts = forecast_next_days(
                ds,
                mask,
                means,
                stds,
                model,
                as_of_date,
                horizon=3,
            )

    except Exception as exc:
        st.error(f"Three-day forecast failed: {exc}")
        st.stop()

    # --------------------------------------------------------
    # Regional forecast summary table
    # --------------------------------------------------------

    st.subheader("Regional forecast summary")

    summary_rows = []

    for item in forecasts:
        regional = item["regional_means"]

        summary_rows.append(
            {
                "Forecast date": pd.Timestamp(
                    item["date"]
                ).strftime("%d %b %Y"),
                "Rainfall (mm/day)": regional["rainfall"],
                "Tmax (°C)": regional["tmax"],
                "Tmin (°C)": regional["tmin"],
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    st.dataframe(
        summary_df.round(2),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Metrics for each forecast day
    # --------------------------------------------------------

    st.subheader("Daily forecast details")

    for day_number, item in enumerate(
        forecasts,
        start=1,
    ):
        forecast_date = pd.Timestamp(item["date"])
        regional = item["regional_means"]

        st.markdown(
            f"### Day {day_number}: "
            f"{forecast_date:%A, %d %B %Y}"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted rainfall",
            f"{regional['rainfall']:.2f} mm/day",
        )

        c2.metric(
            "Predicted Tmax",
            f"{regional['tmax']:.2f} °C",
        )

        c3.metric(
            "Predicted Tmin",
            f"{regional['tmin']:.2f} °C",
        )

    # --------------------------------------------------------
    # Select which forecast day to display on the map
    # --------------------------------------------------------

    st.subheader("Forecast map and grid-cell details")

    map_day_options = {
        pd.Timestamp(item["date"]).strftime("%d %b %Y"): i
        for i, item in enumerate(forecasts)
    }

    selected_map_day = st.selectbox(
        "Choose forecast day to display on the map",
        options=list(map_day_options.keys()),
        key="forecast_map_day",
    )

    selected_item = forecasts[
        map_day_options[selected_map_day]
    ]

    pred_grid = selected_item["grid"]

    forecast_date = pd.Timestamp(
        selected_item["date"]
    )

    variable = st.selectbox(
        "Forecast map layer",
        ["Rainfall", "Tmax", "Tmin"],
        key="forecast_layer",
    )

    grid_df = build_grid_table(
        ds,
        mask,
        pred_grid,
    )

    col = {
        "Rainfall": "rainfall",
        "Tmax": "tmax",
        "Tmin": "tmin",
    }[variable]

    map_obj = folium.Map(
        location=[24.1, 88.0],
        zoom_start=7,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    add_map_layers(
        map_obj,
        grid_df,
        col,
        variable,
    )

    st_folium(
        map_obj,
        use_container_width=True,
        height=590,
        returned_objects=[],
    )

    # --------------------------------------------------------
    # Grid-cell table for the selected forecast day
    # --------------------------------------------------------

    st.subheader(
        f"Grid-cell forecast values — {forecast_date:%d %b %Y}"
    )

    display_cols = [
        "reference_name",
        "latitude",
        "longitude",
        "rainfall",
        "tmax",
        "tmin",
    ]

    st.dataframe(
        grid_df[display_cols].round(2),
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Download the selected forecast day's grid as CSV
    # --------------------------------------------------------

    csv_data = grid_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download selected day's forecast as CSV",
        data=csv_data,
        file_name=(
            f"varuna_forecast_{forecast_date:%Y-%m-%d}.csv"
        ),
        mime="text/csv",
    )

    st.info(
        "Forecasts for Days 2 and 3 are recursive: earlier "
        "predictions are fed into later forecast steps. "
        "Forecast errors may accumulate with lead time. "
        "These forecasts are experimental and are not a "
        "validated operational weather warning."
    )


# ============================================================
# PAGE 3: WHAT-IF SIMULATOR
# ============================================================

elif page == "What-if Simulator":

    st.subheader("What-if simulator")

    st.write(
        "Perturb the final observed day in the seven-day "
        "input window and compare the model response."
    )

    st.caption(
        f"Simulation target date: {selected_target}"
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        rain_delta = st.slider(
            "Rainfall change (mm)",
            -100.0,
            100.0,
            0.0,
            5.0,
        )

    with c2:
        tmax_delta = st.slider(
            "Tmax change (°C)",
            -5.0,
            5.0,
            0.0,
            0.5,
        )

    with c3:
        tmin_delta = st.slider(
            "Tmin change (°C)",
            -5.0,
            5.0,
            0.0,
            0.5,
        )

    if st.button(
        "Run baseline vs scenario",
        type="primary",
    ):
        from src.forecasting import run_scenario

        try:
            (
                baseline,
                scenario,
                baseline_grid,
                scenario_grid,
            ) = run_scenario(
                ds,
                mask,
                means,
                stds,
                model,
                selected_target,
                rain_delta,
                tmax_delta,
                tmin_delta,
            )

        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.stop()

        rows = []

        for var, label, unit in [
            ("rainfall", "Rainfall", "mm/day"),
            ("tmax", "Tmax", "°C"),
            ("tmin", "Tmin", "°C"),
        ]:
            rows.append(
                {
                    "Variable": f"{label} ({unit})",
                    "Baseline": baseline[var],
                    "Scenario": scenario[var],
                    "Difference": (
                        scenario[var] - baseline[var]
                    ),
                }
            )

        comp = pd.DataFrame(rows)

        st.subheader("Baseline versus scenario")

        st.dataframe(
            comp.round(2),
            use_container_width=True,
            hide_index=True,
        )

        chart = comp.melt(
            id_vars="Variable",
            value_vars=["Baseline", "Scenario"],
            var_name="Run",
            value_name="Predicted value",
        )

        st.plotly_chart(
            px.bar(
                chart,
                x="Variable",
                y="Predicted value",
                color="Run",
                barmode="group",
            ),
            use_container_width=True,
        )

        # Map the difference between the scenario and baseline.
        map_variable = st.selectbox(
            "Impact map variable",
            ["Rainfall", "Tmax", "Tmin"],
            key="impact_map",
        )

        key = {
            "Rainfall": 0,
            "Tmax": 1,
            "Tmin": 2,
        }[map_variable]

        delta_grid = (
            scenario_grid[:, :, key]
            - baseline_grid[:, :, key]
        )

        delta_table = build_grid_table(
            ds,
            mask,
            delta_grid[:, :, None],
            one_channel=True,
        )

        delta_table["impact"] = delta_grid[mask]

        map_obj = folium.Map(
            location=[24.1, 88.0],
            zoom_start=7,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        add_map_layers(
            map_obj,
            delta_table,
            "impact",
            f"Scenario impact: {map_variable}",
        )

        st_folium(
            map_obj,
            use_container_width=True,
            height=560,
            returned_objects=[],
        )

        st.caption(
            "This shows model sensitivity to altered inputs. "
            "It is not a physically validated causal simulation."
        )


# ============================================================
# PAGE 4: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.subheader("Model performance")

    st.write(
        "The model was trained using a chronological split: "
        "training through 2022, validation in 2023–2024, "
        "and test data in 2025."
    )

    metrics_path = (
        ROOT
        / "models"
        / "convlstm_attention_1day_metrics.csv"
    )

    if metrics_path.exists():

        metrics = pd.read_csv(metrics_path)

        st.dataframe(
            metrics,
            use_container_width=True,
            hide_index=True,
        )

        numeric = [
            c
            for c in metrics.columns
            if c.lower() in {"mae", "rmse", "r2", "mse"}
        ]

        if numeric:
            st.bar_chart(
                metrics.set_index(
                    metrics.columns[0]
                )[numeric]
            )

    else:
        st.info(
            "No `models/convlstm_attention_1day_metrics.csv` "
            "file was found. This page will not invent metrics. "
            "Export the real test-set metrics from your "
            "evaluation notebook to this path to display them here."
        )

    st.markdown("#### What the metrics mean")

    st.markdown(
        """
        - **MAE**: average absolute forecast error, in the
          variable's original units.
        - **RMSE**: gives larger errors more influence.
        - **Forecast vs observation**: compare predictions
          with recorded values for dates where observations exist.
        """
    )

    st.warning(
        "The selected 2025 test period should remain the final "
        "evaluation period; avoid tuning model choices against "
        "it repeatedly."
    )


# ============================================================
# PAGE 5: ABOUT VARUNA
# ============================================================

else:

    st.subheader("About VARUNA")

    st.markdown(
        """
        **VARUNA** is an AI-assisted climate exploration
        prototype for West Bengal.

        **Data currently used**

        - Daily IMD rainfall, maximum temperature (Tmax),
          and minimum temperature (Tmin)
        - Historical period in the prepared NetCDF file
        - 14 selected grid cells with complete observations
          in the prepared study mask
        - Rainfall and temperature represented on the same
          7 × 5 grid

        **Forecast model**

        - ConvLSTM with temporal attention
        - Seven historical days as input
        - One-day prediction for each grid cell
        - Recursive three-day forecasts are generated by
          feeding previous predictions into subsequent steps
        - Forecast outputs are shown in physical units after
          inverse normalization

        **Current limitations**

        - This local prototype uses the prepared historical
          archive; it is not a live operational forecast service.
        - Grid-cell centres are coarse model-grid locations,
          not weather-station coordinates.
        - Place labels on the map are approximate reference
          labels for grid cells.
        - Recursive forecast errors may accumulate with lead time.
        - Risk alerts, uncertainty intervals, and causal scenario
          claims are not yet validated operational products.
        """
    )

