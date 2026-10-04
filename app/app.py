
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.evaluation import (
    inspect_test_data,
    evaluate_all_models
)

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
        "Risk Alerts",
        "Temperature Anomalies",
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

# elif page == "What-if Simulator":

#     st.subheader("What-if simulator")

#     st.write(
#         "Perturb the final observed day in the seven-day "
#         "input window and compare the model response."
#     )

#     st.caption(
#         f"Simulation target date: {selected_target}"
#     )

#     c1, c2, c3 = st.columns(3)

#     with c1:
#         rain_delta = st.slider(
#             "Rainfall change (mm)",
#             -100.0,
#             100.0,
#             0.0,
#             5.0,
#         )

#     with c2:
#         tmax_delta = st.slider(
#             "Tmax change (°C)",
#             -5.0,
#             5.0,
#             0.0,
#             0.5,
#         )

#     with c3:
#         tmin_delta = st.slider(
#             "Tmin change (°C)",
#             -5.0,
#             5.0,
#             0.0,
#             0.5,
#         )

#     if st.button(
#         "Run baseline vs scenario",
#         type="primary",
#     ):
#         from src.forecasting import run_scenario

#         try:
#             (
#                 baseline,
#                 scenario,
#                 baseline_grid,
#                 scenario_grid,
#             ) = run_scenario(
#                 ds,
#                 mask,
#                 means,
#                 stds,
#                 model,
#                 selected_target,
#                 rain_delta,
#                 tmax_delta,
#                 tmin_delta,
#             )

#         except Exception as exc:
#             st.error(f"Simulation failed: {exc}")
#             st.stop()

#         rows = []

#         for var, label, unit in [
#             ("rainfall", "Rainfall", "mm/day"),
#             ("tmax", "Tmax", "°C"),
#             ("tmin", "Tmin", "°C"),
#         ]:
#             rows.append(
#                 {
#                     "Variable": f"{label} ({unit})",
#                     "Baseline": baseline[var],
#                     "Scenario": scenario[var],
#                     "Difference": (
#                         scenario[var] - baseline[var]
#                     ),
#                 }
#             )

#         comp = pd.DataFrame(rows)

#         st.subheader("Baseline versus scenario")

#         st.dataframe(
#             comp.round(2),
#             use_container_width=True,
#             hide_index=True,
#         )

#         chart = comp.melt(
#             id_vars="Variable",
#             value_vars=["Baseline", "Scenario"],
#             var_name="Run",
#             value_name="Predicted value",
#         )

#         st.plotly_chart(
#             px.bar(
#                 chart,
#                 x="Variable",
#                 y="Predicted value",
#                 color="Run",
#                 barmode="group",
#             ),
#             use_container_width=True,
#         )

#         # Map the difference between the scenario and baseline.
#         map_variable = st.selectbox(
#             "Impact map variable",
#             ["Rainfall", "Tmax", "Tmin"],
#             key="impact_map",
#         )

#         key = {
#             "Rainfall": 0,
#             "Tmax": 1,
#             "Tmin": 2,
#         }[map_variable]

#         delta_grid = (
#             scenario_grid[:, :, key]
#             - baseline_grid[:, :, key]
#         )

#         delta_table = build_grid_table(
#             ds,
#             mask,
#             delta_grid[:, :, None],
#             one_channel=True,
#         )

#         delta_table["impact"] = delta_grid[mask]

#         map_obj = folium.Map(
#             location=[24.1, 88.0],
#             zoom_start=7,
#             tiles="OpenStreetMap",
#             control_scale=True,
#         )

#         add_map_layers(
#             map_obj,
#             delta_table,
#             "impact",
#             f"Scenario impact: {map_variable}",
#         )

#         st_folium(
#             map_obj,
#             use_container_width=True,
#             height=560,
#             returned_objects=[],
#         )

#         st.caption(
#             "This shows model sensitivity to altered inputs. "
#             "It is not a physically validated causal simulation."
#         )




elif page == "What-if Simulator":

    from src.forecasting import run_scenario, GRID_REFERENCE_NAMES

    st.subheader("Location-based what-if simulator")

    st.write(
        "Choose one available grid cell, modify its latest observed "
        "rainfall or temperature, and compare the model's prediction "
        "with the baseline forecast."
    )

    # --------------------------------------------------------
    # Build a selector from valid grid cells only
    # --------------------------------------------------------
    locations = {}

    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            if not mask[row, col]:
                continue

            lat = float(ds.latitude.values[row])
            lon = float(ds.longitude.values[col])

            reference_name, _ = GRID_REFERENCE_NAMES.get(
                (round(lat, 1), round(lon, 1)),
                (f"Grid cell {lat:.1f}, {lon:.1f}", "Grid cell"),
            )

            label = (
                f"{reference_name} "
                f"({lat:.1f}°N, {lon:.1f}°E)"
            )
            locations[label] = (row, col)

    selected_location_label = st.selectbox(
        "Select a location",
        options=list(locations.keys()),
        key="whatif_location",
    )

    row, col = locations[selected_location_label]

    target_timestamp = pd.Timestamp(selected_target).normalize()
    dataset_dates = pd.to_datetime(ds.time.values).normalize()
    prior_indices = np.flatnonzero(dataset_dates < target_timestamp)

    if len(prior_indices) < 7:
        st.error("This target date does not have seven prior observations.")
        st.stop()

    latest_idx = int(prior_indices[-1])
    latest_date = dataset_dates[latest_idx]

    # Read the latest observed values at this particular cell.
    current_rain = float(
        ds["rainfall"].isel(time=latest_idx).values[row, col]
    )
    current_tmax = float(
        ds["tmax"].isel(time=latest_idx).values[row, col]
    )
    current_tmin = float(
        ds["tmin"].isel(time=latest_idx).values[row, col]
    )

    if not np.isfinite([current_rain, current_tmax, current_tmin]).all():
        st.error(
            "One or more climate observations are missing for this "
            "location and date. Choose another location or target date."
        )
        st.stop()

    st.caption(
        f"Selected grid cell: {selected_location_label}  |  "
        f"Latest observation: {latest_date:%d %b %Y}  |  "
        f"Forecast target: {target_timestamp:%d %b %Y}"
    )

    st.markdown("### Latest observed values")

    obs1, obs2, obs3 = st.columns(3)
    obs1.metric("Rainfall", f"{current_rain:.2f} mm/day")
    obs2.metric("Maximum temperature", f"{current_tmax:.2f} °C")
    obs3.metric("Minimum temperature", f"{current_tmin:.2f} °C")

    st.markdown("### Modify the selected location")

    st.caption(
        "The sliders apply changes to this grid cell on the final day "
        "of the model's seven-day input window. Other grid cells are "
        "left unchanged."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        rain_delta = st.slider(
            "Rainfall change (mm)",
            min_value=-100.0,
            max_value=100.0,
            value=0.0,
            step=1.0,
            key="whatif_rain_delta",
        )

    with c2:
        tmax_delta = st.slider(
            "Maximum temperature change (°C)",
            min_value=-15.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
            key="whatif_tmax_delta",
        )

    with c3:
        tmin_delta = st.slider(
            "Minimum temperature change (°C)",
            min_value=-15.0,
            max_value=15.0,
            value=0.0,
            step=0.5,
            key="whatif_tmin_delta",
        )

    scenario_rain = max(0.0, current_rain + rain_delta)
    scenario_tmax = current_tmax + tmax_delta
    scenario_tmin = current_tmin + tmin_delta

    st.markdown("### Input comparison")

    input_comparison = pd.DataFrame(
        [
            {
                "Variable": "Rainfall (mm/day)",
                "Observed value": current_rain,
                "Scenario input": scenario_rain,
                "Applied change": scenario_rain - current_rain,
            },
            {
                "Variable": "Maximum temperature (°C)",
                "Observed value": current_tmax,
                "Scenario input": scenario_tmax,
                "Applied change": scenario_tmax - current_tmax,
            },
            {
                "Variable": "Minimum temperature (°C)",
                "Observed value": current_tmin,
                "Scenario input": scenario_tmin,
                "Applied change": scenario_tmin - current_tmin,
            },
        ]
    )

    st.dataframe(
        input_comparison.round(2),
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "Run location-based simulation",
        type="primary",
        key="run_location_whatif",
    ):
        try:
            with st.spinner("Running baseline and scenario predictions..."):
                baseline, scenario, baseline_grid, scenario_grid = run_scenario(
                    ds=ds,
                    mask=mask,
                    means=means,
                    stds=stds,
                    model=model,
                    target_date=selected_target,
                    rainfall_change=rain_delta,
                    tmax_change=tmax_delta,
                    tmin_change=tmin_delta,
                    location_index=(row, col),
                )

            rows = []

            for var, label, unit in [
                ("rainfall", "Rainfall", "mm/day"),
                ("tmax", "Maximum temperature", "°C"),
                ("tmin", "Minimum temperature", "°C"),
            ]:
                base_value = baseline[var]
                scenario_value = scenario[var]

                rows.append(
                    {
                        "Predicted variable": f"{label} ({unit})",
                        "Baseline prediction": base_value,
                        "Scenario prediction": scenario_value,
                        "Difference": scenario_value - base_value,
                    }
                )

            result_df = pd.DataFrame(rows)

            st.markdown("### Forecast comparison for selected location")

            m1, m2, m3 = st.columns(3)

            m1.metric(
                "Scenario rainfall",
                f"{scenario['rainfall']:.2f} mm/day",
                delta=f"{scenario['rainfall'] - baseline['rainfall']:+.2f}",
            )
            m2.metric(
                "Scenario Tmax",
                f"{scenario['tmax']:.2f} °C",
                delta=f"{scenario['tmax'] - baseline['tmax']:+.2f} °C",
            )
            m3.metric(
                "Scenario Tmin",
                f"{scenario['tmin']:.2f} °C",
                delta=f"{scenario['tmin'] - baseline['tmin']:+.2f} °C",
            )

            st.dataframe(
                result_df.round(2),
                use_container_width=True,
                hide_index=True,
            )

            chart_df = result_df.set_index("Predicted variable")[
                ["Baseline prediction", "Scenario prediction"]
            ]
            st.bar_chart(chart_df)

            st.success(
                "Simulation completed for the selected grid cell. "
                "The baseline uses the original input observations; "
                "the scenario changes only the selected cell."
            )

        except Exception as exc:
            st.error(f"Simulation failed: {exc}")

    st.info(
        "This experiment measures the model's sensitivity to modified "
        "inputs. It does not establish that changing local weather "
        "conditions would physically cause the predicted outcome. "
        "Grid cells represent coarse model locations, not weather stations."
    )




# ============================================================
# PAGE5: COMBINED FLOOD AND DROUGHT RISK ALERTS
# ============================================================

elif page == "Risk Alerts":

    st.subheader("Combined Flood and Drought Risk Alerts")

    st.write(
        "View combined flood and drought risk levels, "
        "hazard classifications, alert priorities, and "
        "affected grid locations across West Bengal."
    )

    # --------------------------------------------------------
    # STEP 1: Define the paths to the generated output files
    # --------------------------------------------------------

    ALERTS_DIR = ROOT / "results" / "risk_alerts"

    COMBINED_PATH = ALERTS_DIR / "combined_risk_alerts.csv"
    LATEST_PATH = ALERTS_DIR / "latest_risk_alerts.csv"
    SUMMARY_PATH = ALERTS_DIR / "dashboard_risk_summary.json"

    # --------------------------------------------------------
    # STEP 2: Load the generated CSV files
    # --------------------------------------------------------

    @st.cache_data
    def load_risk_alert_files(combined_path, latest_path):
        combined = pd.read_csv(combined_path)
        latest = pd.read_csv(latest_path)

        combined["date"] = pd.to_datetime(
            combined["date"], errors="coerce"
        )
        latest["date"] = pd.to_datetime(
            latest["date"], errors="coerce"
        )

        return combined, latest

    if not COMBINED_PATH.exists() or not LATEST_PATH.exists():
        st.error(
            "Risk alert output files were not found. "
            "Run the combined risk alerts module first."
        )
        st.code(
            "python src/risk_alerts/risk_alerts_combined.py"
        )
        st.stop()

    try:
        combined_df, latest_df = load_risk_alert_files(
            str(COMBINED_PATH),
            str(LATEST_PATH),
        )

    except Exception as exc:
        st.error(f"Could not load risk alert files: {exc}")
        st.stop()

    # --------------------------------------------------------
    # STEP 3: Validate the required columns
    # --------------------------------------------------------

    required_columns = {
        "date",
        "latitude",
        "longitude",
        "flood_risk_level",
        "drought_risk_level",
        "combined_risk_level",
        "hazard_type",
        "alert_priority",
        "alert_message",
    }

    missing = required_columns - set(combined_df.columns)

    if missing:
        st.error(
            "The combined alerts CSV is missing required "
            f"columns: {', '.join(sorted(missing))}"
        )
        st.stop()

    if combined_df.empty:
        st.warning("The combined risk alerts file is empty.")
        st.stop()

    # Remove records without usable dates or coordinates.
    combined_df = combined_df.dropna(
        subset=["date", "latitude", "longitude"]
    ).copy()

    latest_df = latest_df.dropna(
        subset=["date", "latitude", "longitude"]
    ).copy()

    if combined_df.empty:
        st.warning("No valid dated risk observations are available.")
        st.stop()

    # --------------------------------------------------------
    # STEP 4: Display the latest date and summary metrics
    # --------------------------------------------------------

    latest_date = combined_df["date"].max()

    # Use the generated latest-date file for current alerts.
    current_alerts = latest_df[
        latest_df["date"] == latest_df["date"].max()
    ].copy() if not latest_df.empty else latest_df.copy()

    st.caption(
        f"Latest observation date: {latest_date:%d %B %Y}"
    )

    total_observations = len(combined_df)

    high_extreme_count = int(
        combined_df["combined_risk_level"]
        .astype(str)
        .str.upper()
        .isin(["HIGH", "EXTREME"])
        .sum()
    )

    extreme_count = int(
        combined_df["combined_risk_level"]
        .astype(str)
        .str.upper()
        .eq("EXTREME")
        .sum()
    )

    critical_count = int(
        current_alerts["alert_priority"]
        .astype(str)
        .str.upper()
        .eq("CRITICAL")
        .sum()
    ) if "alert_priority" in current_alerts.columns else 0

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total observations", f"{total_observations:,}")
    c2.metric("High / Extreme observations", f"{high_extreme_count:,}")
    c3.metric("Extreme observations", f"{extreme_count:,}")
    c4.metric("Critical latest alerts", f"{critical_count:,}")

    st.caption(
        "These counts represent records in the generated dataset, "
        "not necessarily unique locations or independent events."
    )

    # --------------------------------------------------------
    # STEP 5: Combined risk distribution
    # --------------------------------------------------------

    st.markdown("### Combined risk distribution")

    risk_order = ["LOW", "MODERATE", "HIGH", "EXTREME"]

    risk_counts = (
        combined_df["combined_risk_level"]
        .astype(str)
        .str.upper()
        .value_counts()
        .reindex(risk_order, fill_value=0)
        .rename_axis("Risk level")
        .reset_index(name="Observations")
    )

    c1, c2 = st.columns([1, 1])

    with c1:
        fig_risk = px.bar(
            risk_counts,
            x="Risk level",
            y="Observations",
            category_orders={"Risk level": risk_order},
            title="Observations by combined risk level",
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    with c2:
        fig_hazard = px.pie(
            combined_df,
            names="hazard_type",
            title="Hazard type distribution",
        )
        st.plotly_chart(fig_hazard, use_container_width=True)

    # --------------------------------------------------------
    # STEP 6: Filter the alerts shown to the user
    # --------------------------------------------------------

    st.markdown("### Explore risk alerts")

    available_levels = [
        level for level in risk_order
        if level in combined_df["combined_risk_level"]
        .astype(str).str.upper().unique()
    ]

    selected_levels = st.multiselect(
        "Filter by combined risk level",
        options=available_levels,
        default=["HIGH", "EXTREME"],
    )

    available_hazards = sorted(
        combined_df["hazard_type"]
        .dropna().astype(str).unique().tolist()
    )

    selected_hazards = st.multiselect(
        "Filter by hazard type",
        options=available_hazards,
        default=available_hazards,
    )

    min_date = combined_df["date"].min().date()
    max_date = combined_df["date"].max().date()

    date_range = st.date_input(
        "Filter by observation date",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    filtered_df = combined_df.copy()

    filtered_df = filtered_df[
        filtered_df["combined_risk_level"]
        .astype(str).str.upper().isin(selected_levels)
    ]

    filtered_df = filtered_df[
        filtered_df["hazard_type"]
        .astype(str).isin(selected_hazards)
    ]

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            filtered_df["date"].dt.date.between(
                start_date, end_date
            )
        ]

    st.caption(f"{len(filtered_df):,} records match your filters.")

    # --------------------------------------------------------
    # STEP 7: Display alert records in a table
    # --------------------------------------------------------

    display_columns = [
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

    st.dataframe(
        filtered_df[display_columns]
        .sort_values(
            ["date", "combined_risk_level"],
            ascending=[False, False],
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_download = filtered_df[display_columns].to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "Download filtered risk alerts",
        data=csv_download,
        file_name="varuna_filtered_risk_alerts.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # STEP 8: Plot the filtered alerts on a spatial map
    # --------------------------------------------------------

    st.markdown("### Spatial distribution of risk alerts")

    if filtered_df.empty:
        st.info("No alerts match the selected filters.")
    else:
        map_df = filtered_df.copy()

        map_df["combined_risk_level"] = (
            map_df["combined_risk_level"]
            .astype(str).str.upper()
        )

        # Keep the latest record per grid cell for the map.
        map_df = (
            map_df.sort_values("date")
            .drop_duplicates(
                subset=["latitude", "longitude"],
                keep="last",
            )
        )

        risk_colors = {
            "LOW": "green",
            "MODERATE": "blue",
            "HIGH": "orange",
            "EXTREME": "red",
        }

        map_obj = folium.Map(
            location=[24.1, 88.0],
            zoom_start=7,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        for _, row in map_df.iterrows():
            level = row["combined_risk_level"]
            color = risk_colors.get(level, "gray")

            popup_html = (
                f"<b>Combined risk:</b> {level}<br>"
                f"<b>Date:</b> {row['date']:%Y-%m-%d}<br>"
                f"<b>Flood:</b> {row['flood_risk_level']}<br>"
                f"<b>Drought:</b> {row['drought_risk_level']}<br>"
                f"<b>Hazard:</b> {row['hazard_type']}<br>"
                f"<b>Priority:</b> {row['alert_priority']}<br>"
                f"<b>Alert:</b> {row['alert_message']}"
            )

            folium.CircleMarker(
                location=[
                    row["latitude"],
                    row["longitude"],
                ],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=f"{level} risk",
                popup=folium.Popup(
                    popup_html,
                    max_width=350,
                ),
            ).add_to(map_obj)

        st_folium(
            map_obj,
            use_container_width=True,
            height=570,
            returned_objects=[],
        )

        st.caption(
            "The map shows the latest matching record per grid cell. "
            "Grid coordinates are coarse model-grid centres, "
            "not weather-station locations."
        )

    # --------------------------------------------------------
    # STEP 9: Show the current latest-date alert table
    # --------------------------------------------------------

    st.markdown("### Latest available alerts")

    if current_alerts.empty:
        st.info("No records are available in the latest alerts file.")
    else:
        st.dataframe(
            current_alerts[
                [
                    col for col in display_columns
                    if col in current_alerts.columns
                ]
            ].sort_values(
                "combined_risk_level",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

    # --------------------------------------------------------
    # STEP 10: Optional JSON summary availability
    # --------------------------------------------------------

    if SUMMARY_PATH.exists():
        with st.expander("View dashboard summary JSON"):
            try:
                with open(
                    SUMMARY_PATH, "r", encoding="utf-8"
                ) as summary_file:
                    summary_data = json.load(summary_file)

                st.json(summary_data)

            except (OSError, json.JSONDecodeError) as exc:
                st.warning(f"Could not read summary JSON: {exc}")


# ============================================================
# PAGE: TEMPERATURE ANOMALIES
# ============================================================

elif page == "Temperature Anomalies":

    st.subheader("Temperature Anomaly Detection")
    st.write(
        "Explore unusually hot and cold maximum-temperature "
        "observations across the West Bengal study grid, "
        "relative to the historical day-of-year climatology."
    )

    # --------------------------------------------------------
    # STEP 1: Define output paths
    # --------------------------------------------------------

    ANOMALY_DIR = ROOT / "results" / "anomaly"

    HOT_PATH = ANOMALY_DIR / "strongest_hot_anomalies.csv"
    COLD_PATH = ANOMALY_DIR / "strongest_cold_anomalies.csv"
    ALL_PATH = ANOMALY_DIR / "tmax_anomaly_records.csv"

    # --------------------------------------------------------
    # STEP 2: Load anomaly outputs
    # --------------------------------------------------------

    @st.cache_data
    def load_anomaly_files(all_path, hot_path, cold_path):
        all_records = pd.read_csv(all_path)
        hot_records = pd.read_csv(hot_path)
        cold_records = pd.read_csv(cold_path)

        for df in [all_records, hot_records, cold_records]:
            if "date" in df.columns:
                df["date"] = pd.to_datetime(
                    df["date"], errors="coerce"
                )

        return all_records, hot_records, cold_records

    missing_files = [
        path.name
        for path in [ALL_PATH, HOT_PATH, COLD_PATH]
        if not path.exists()
    ]

    if missing_files:
        st.error(
            "The following anomaly output files are missing: "
            + ", ".join(missing_files)
        )
        st.info(
            "Run the anomaly-detection notebook or its "
            "equivalent pipeline to generate the CSV files."
        )
        st.stop()

    try:
        all_df, hot_df, cold_df = load_anomaly_files(
            str(ALL_PATH),
            str(HOT_PATH),
            str(COLD_PATH),
        )
    except Exception as exc:
        st.error(f"Could not load anomaly records: {exc}")
        st.stop()

    # --------------------------------------------------------
    # STEP 3: Validate the full anomaly dataset
    # --------------------------------------------------------

    required = {
        "date",
        "latitude",
        "longitude",
        "tmax",
        "normal_tmax",
        "anomaly_celsius",
        "zscore",
        "temperature_status",
    }

    missing_columns = required - set(all_df.columns)

    if missing_columns:
        st.error(
            "The full anomaly CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )
        st.stop()

    all_df = all_df.dropna(
        subset=["date", "latitude", "longitude", "zscore"]
    ).copy()

    all_df["temperature_status"] = (
        all_df["temperature_status"].astype(str).str.upper()
    )

    if all_df.empty:
        st.warning("No valid anomaly records are available.")
        st.stop()

    # --------------------------------------------------------
    # STEP 4: Summary metrics
    # --------------------------------------------------------

    latest_date = all_df["date"].max()

    hot_count = int(
        all_df["temperature_status"].eq("HOT_ANOMALY").sum()
    )
    cold_count = int(
        all_df["temperature_status"].eq("COLD_ANOMALY").sum()
    )
    normal_count = int(
        all_df["temperature_status"].eq("NORMAL").sum()
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Hot anomaly records", f"{hot_count:,}")
    c2.metric("Cold anomaly records", f"{cold_count:,}")
    c3.metric("Normal records", f"{normal_count:,}")
    c4.metric("Latest observation", latest_date.strftime("%d %b %Y"))

    st.caption(
        "Anomalies are identified using Tmax z-scores: "
        "hot ≥ +2 and cold ≤ −2. Counts represent grid-cell "
        "observations across the full historical archive."
    )

    # --------------------------------------------------------
    # STEP 5: Filter by date and anomaly type
    # --------------------------------------------------------

    st.markdown("### Explore historical anomalies")

    statuses = ["HOT_ANOMALY", "COLD_ANOMALY", "NORMAL"]

    selected_statuses = st.multiselect(
        "Anomaly classification",
        options=statuses,
        default=["HOT_ANOMALY", "COLD_ANOMALY"],
    )

    min_date = all_df["date"].min().date()
    max_date = all_df["date"].max().date()

    date_range = st.date_input(
        "Observation date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="anomaly_date_range",
    )

    filtered = all_df[
        all_df["temperature_status"].isin(selected_statuses)
    ].copy()

    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            filtered["date"].dt.date.between(
                start_date, end_date
            )
        ]

    st.caption(f"{len(filtered):,} records match your filters.")

    # --------------------------------------------------------
    # STEP 6: Visualize anomaly classifications
    # --------------------------------------------------------

    st.markdown("### Anomaly distribution")

    status_counts = (
        all_df["temperature_status"]
        .value_counts()
        .reindex(statuses, fill_value=0)
        .rename_axis("Classification")
        .reset_index(name="Observations")
    )

    left, right = st.columns(2)

    with left:
        fig_status = px.bar(
            status_counts,
            x="Classification",
            y="Observations",
            title="Historical Tmax anomaly classifications",
            color="Classification",
            color_discrete_map={
                "HOT_ANOMALY": "#D94841",
                "COLD_ANOMALY": "#3182BD",
                "NORMAL": "#718096",
            },
        )
        st.plotly_chart(fig_status, use_container_width=True)

    with right:
        if not filtered.empty:
            fig_zscore = px.histogram(
                filtered,
                x="zscore",
                color="temperature_status",
                nbins=40,
                title="Distribution of Tmax anomaly z-scores",
                color_discrete_map={
                    "HOT_ANOMALY": "#D94841",
                    "COLD_ANOMALY": "#3182BD",
                    "NORMAL": "#718096",
                },
            )
            st.plotly_chart(fig_zscore, use_container_width=True)
        else:
            st.info("No records match the selected filters.")

    # --------------------------------------------------------
    # STEP 7: Map the latest matching record per grid cell
    # --------------------------------------------------------

    st.markdown("### Spatial distribution of anomalies")

    if filtered.empty:
        st.info("No anomaly records match the selected filters.")
    else:
        map_df = (
            filtered.sort_values("date")
            .drop_duplicates(
                subset=["latitude", "longitude"],
                keep="last",
            )
        )

        anomaly_colors = {
            "HOT_ANOMALY": "red",
            "COLD_ANOMALY": "blue",
            "NORMAL": "gray",
        }

        anomaly_map = folium.Map(
            location=[24.1, 88.0],
            zoom_start=7,
            tiles="OpenStreetMap",
            control_scale=True,
        )

        for _, row in map_df.iterrows():
            status = row["temperature_status"]
            color = anomaly_colors.get(status, "gray")

            popup = (
                f"<b>Status:</b> {status}<br>"
                f"<b>Date:</b> {row['date']:%Y-%m-%d}<br>"
                f"<b>Observed Tmax:</b> {row['tmax']:.2f} °C<br>"
                f"<b>Normal Tmax:</b> {row['normal_tmax']:.2f} °C<br>"
                f"<b>Anomaly:</b> {row['anomaly_celsius']:.2f} °C<br>"
                f"<b>Z-score:</b> {row['zscore']:.2f}"
            )

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                tooltip=status.replace("_", " "),
                popup=folium.Popup(popup, max_width=300),
            ).add_to(anomaly_map)

        st_folium(
            anomaly_map,
            use_container_width=True,
            height=550,
            returned_objects=[],
        )

        st.caption(
            "The map shows the latest matching record for each "
            "grid cell within the selected date range. Coordinates "
            "represent coarse grid-cell centres."
        )

    # --------------------------------------------------------
    # STEP 8: Show the strongest hot and cold anomalies
    # --------------------------------------------------------

    st.markdown("### Strongest historical anomalies")

    hot_tab, cold_tab = st.tabs(
        ["Strongest hot anomalies", "Strongest cold anomalies"]
    )

    anomaly_columns = [
        "date",
        "latitude",
        "longitude",
        "tmax",
        "normal_tmax",
        "anomaly_celsius",
        "zscore",
        "temperature_status",
    ]

    with hot_tab:
        if hot_df.empty:
            st.info("No strongest-hot records are available.")
        else:
            hot_display = hot_df.copy()
            if "date" in hot_display.columns:
                hot_display["date"] = pd.to_datetime(
                    hot_display["date"], errors="coerce"
                )
            cols = [
                col for col in anomaly_columns
                if col in hot_display.columns
            ]
            st.dataframe(
                hot_display[cols],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download strongest hot anomalies",
                data=hot_display.to_csv(index=False).encode("utf-8"),
                file_name="strongest_hot_anomalies.csv",
                mime="text/csv",
            )

    with cold_tab:
        if cold_df.empty:
            st.info("No strongest-cold records are available.")
        else:
            cold_display = cold_df.copy()
            if "date" in cold_display.columns:
                cold_display["date"] = pd.to_datetime(
                    cold_display["date"], errors="coerce"
                )
            cols = [
                col for col in anomaly_columns
                if col in cold_display.columns
            ]
            st.dataframe(
                cold_display[cols],
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download strongest cold anomalies",
                data=cold_display.to_csv(index=False).encode("utf-8"),
                file_name="strongest_cold_anomalies.csv",
                mime="text/csv",
            )

    # --------------------------------------------------------
    # STEP 9: Full filtered anomaly records
    # --------------------------------------------------------

    st.markdown("### Detailed anomaly records")

    detail_columns = [
        col for col in anomaly_columns
        if col in filtered.columns
    ]

    st.dataframe(
        filtered[detail_columns].sort_values(
            "date", ascending=False
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download filtered anomaly records",
        data=filtered[detail_columns]
        .to_csv(index=False)
        .encode("utf-8"),
        file_name="varuna_filtered_tmax_anomalies.csv",
        mime="text/csv",
    )

    st.info(
        "These are historical Tmax anomaly detections based on "
        "the climatology calculated by the anomaly pipeline. "
        "They are not forecasts of future heatwaves or cold waves."
    )

# ============================================================
# PAGE 6: MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.caption(
        "Dynamic evaluation of VARUNA climate forecasting models"
    )

    # ========================================================
    # TEST DATA INFORMATION
    # ========================================================

    st.subheader("Evaluation Configuration")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Forecast Horizon",
        "1 Day"
    )

    c2.metric(
        "Historical Window",
        "7 Days"
    )

    c3.metric(
        "Test Period",
        "2025"
    )

    c4.metric(
        "Spatial Cells",
        "14"
    )

    st.divider()

    # ========================================================
    # TEST DATA INSPECTION
    # ========================================================

    st.subheader("Test Dataset")

    try:

        test_info = inspect_test_data()

        st.dataframe(
            test_info,
            hide_index=True,
            width="stretch"
        )

    except Exception as e:

        st.error(
            f"Unable to load test dataset:\n{e}"
        )

    st.divider()

    # ========================================================
    # MODEL PERFORMANCE
    # ========================================================

    st.subheader(
        "Dynamic Model Evaluation"
    )

    st.info(
        "Metrics are calculated from the saved "
        "2025 test sequences and trained models."
    )

    if st.button(
        "Evaluate Models",
        type="primary"
    ):

        try:

            test_data = np.load(
                Path("data/sequences/test_1day_fixed.npz"),
                allow_pickle=False
            )

            st.write(
                "Available test arrays:",
                test_data.files
            )

            st.warning(
                "The exact input and target arrays "
                "must be mapped to the final model "
                "inputs before evaluation."
            )

        except Exception as e:

            st.error(
                f"Evaluation failed:\n{e}"
            )

# ============================================================
# PAGE 6: ABOUT VARUNA
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

