from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import xarray as xr
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







st.set_page_config(
    page_title="VARUNA | Climate Digital Twin of West Bengal",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual theme ----------
st.markdown("""
<style>
    .block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 16px;
        background: linear-gradient(120deg, #0b1f33 0%, #123f52 55%, #176b70 100%);
        color: white;
        margin-bottom: 1rem;
    }
    .hero h1 {margin: 0; font-size: 2rem;}
    .hero p {margin: .35rem 0 0 0; color: #d7edf0;}
    div[data-testid="stMetric"] {
        background: rgba(120, 150, 160, 0.08);
        border: 1px solid rgba(120, 150, 160, 0.25);
        padding: 12px 14px;
        border-radius: 12px;
    }
    .small-note {font-size: .85rem; color: #718096;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🌦️ VARUNA</h1>
  <p>Climate Digital Twin · West Bengal · IMD daily rainfall, Tmax and Tmin</p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading VARUNA dataset and trained model...")
def resources():
    return load_resources()

try:
    ds, mask, means, stds, model, stats_df = resources()
except Exception as exc:
    st.error("VARUNA could not load its data/model files.")
    st.code(str(exc))
    st.markdown("""
Check that these files exist in the project:
- `data/west_bengal_climate_final.nc`
- `data/normalization_stats.csv`
- `models/varuna_convlstm_attention_1day.keras`

If your model has a different filename, update `MODEL_PATH` in `src/forecasting.py`.
""")
    st.stop()

time_values = pd.to_datetime(ds.time.values)
first_date = time_values[0].date()
last_observed_date = time_values[-1].date()
next_target_date = (time_values[-1] + pd.Timedelta(days=1)).date()

st.sidebar.title("VARUNA controls")
page = st.sidebar.radio(
    "Navigate",
    ["Climate Overview", "Forecast & Spatial Map", "What-if Simulator", "Model Performance", "About VARUNA"],
)
st.sidebar.caption("Data source: the prepared IMD NetCDF file in this project.")
st.sidebar.caption(f"Historical archive: {first_date} — {last_observed_date}")
st.sidebar.caption(f"Selected grid cells: {int(mask.sum())} / {mask.size}")

# Date picker allows historical target dates and the day immediately after the archive.
target_options = get_available_target_dates(ds)
# default_target = next_target_date if next_target_date <= (last_observed_date + pd.Timedelta(days=1)).date() else last_observed_date
default_target = (
    next_target_date
    if next_target_date <= last_observed_date + pd.Timedelta(days=1)
    else last_observed_date
)
selected_target = st.sidebar.date_input(
    "Forecast target date",
    value=default_target,
    min_value=(time_values[7]).date(),
    max_value=next_target_date,
    help="The model uses the seven dates immediately before this target date.",
)

def get_actual_grid(date):
    date64 = np.datetime64(pd.Timestamp(date).to_datetime64())
    hits = np.flatnonzero(ds.time.values.astype("datetime64[ns]") == date64.astype("datetime64[ns]"))
    if len(hits) != 1:
        return None
    idx = int(hits[0])
    return np.stack([ds[v].isel(time=idx).values for v in ["rainfall", "tmax", "tmin"]], axis=-1)

def add_map_layers(map_obj, grid_df, value_col, variable, palette="YlOrRd", show_labels=True):
    # Optional local state boundary. It is the same type of boundary file used during data preparation.
    boundary_path = ROOT / "data" / "state_NWIC.GeoJSON"
    if boundary_path.exists():
        try:
            import json
            with open(boundary_path, "r", encoding="utf-8") as f:
                boundary = json.load(f)
            # If the file contains all states, filter the West Bengal feature.
            features = boundary.get("features", [])
            wb_features = [
                feat for feat in features
                if str(feat.get("properties", {}).get("state_name", "")).lower() == "west bengal"
                or str(feat.get("properties", {}).get("state", "")).lower() in {"west bengal", "west bengal "}
            ]
            geo = {"type": "FeatureCollection", "features": wb_features} if wb_features else boundary
            folium.GeoJson(
                geo,
                name="West Bengal boundary",
                style_function=lambda _: {
                    "color": "#e5f3f5", "weight": 2, "fillColor": "#0b2538", "fillOpacity": 0.08
                },
            ).add_to(map_obj)
        except Exception:
            pass

    if grid_df.empty:
        return

    vals = grid_df[value_col].astype(float).to_numpy()
    finite = vals[np.isfinite(vals)]
    if len(finite) == 0:
        return
    lo, hi = float(finite.min()), float(finite.max())
    if hi == lo:
        hi = lo + 1e-6
    cmap = __import__("branca.colormap", fromlist=["LinearColormap"]).LinearColormap(
        colors=["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"],
        vmin=lo, vmax=hi, caption=f"{variable} ({'mm/day' if variable == 'Rainfall' else '°C'})"
    )
    cmap.add_to(map_obj)

    for _, r in grid_df.iterrows():
        value = float(r[value_col])
        popup = (
            f"<b>{r['reference_name']}</b><br>"
            f"Grid centre: {r['latitude']:.1f}°N, {r['longitude']:.1f}°E<br>"
            f"{variable}: {value:.2f} {'mm/day' if variable == 'Rainfall' else '°C'}"
        )
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=9,
            color="#ffffff",
            weight=1,
            fill=True,
            fill_color=cmap(value),
            fill_opacity=0.92,
            tooltip=f"{r['reference_name']} · {value:.2f}",
            popup=folium.Popup(popup, max_width=300),
        ).add_to(map_obj)
        if show_labels:
            folium.Marker(
                location=[r["latitude"], r["longitude"]],
                icon=folium.DivIcon(
                    html=f"""<div style="font-size:10px;font-weight:700;color:#102a43;
                    text-shadow: 1px 1px 2px white,-1px -1px 2px white;
                    white-space:nowrap; transform:translate(10px,-8px);">
                    {r['short_name']}</div>"""
                ),
            ).add_to(map_obj)
    folium.LayerControl(collapsed=True).add_to(map_obj)

def show_forecast_map(target_date, variable, value_kind="forecast"):
    _, pred_grid = forecast_for_target_date(ds, mask, means, stds, model, target_date)
    grid_df = build_grid_table(ds, mask, pred_grid)
    col = {"Rainfall": "rainfall", "Tmax": "tmax", "Tmin": "tmin"}[variable]
    map_obj = folium.Map(
        location=[24.1, 88.0],
        zoom_start=7,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    add_map_layers(map_obj, grid_df, col, variable)
    st_folium(map_obj, use_container_width=True, height=570, returned_objects=[])
    return grid_df, pred_grid

if page == "Climate Overview":
    st.subheader("Climate overview")
    obs_date = st.date_input(
        "Observed date",
        value=min(last_observed_date, selected_target),
        min_value=first_date,
        max_value=last_observed_date,
        key="overview_date",
    )
    actual = get_actual_grid(obs_date)
    if actual is None:
        st.warning("No observed data exists for that date in the dataset.")
        st.stop()

    regional = regional_means(actual, mask)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regional rainfall", f"{regional['rainfall']:.2f} mm/day")
    c2.metric("Regional Tmax", f"{regional['tmax']:.2f} °C")
    c3.metric("Regional Tmin", f"{regional['tmin']:.2f} °C")
    c4.metric("Valid grid cells", f"{int(mask.sum())} / {mask.size}")

    st.caption("Regional values are unweighted means over the 14 selected grid cells.")
    variable = st.selectbox("Map layer", ["Rainfall", "Tmax", "Tmin"], key="overview_layer")
    grid_df = build_grid_table(ds, mask, actual)
    col = {"Rainfall": "rainfall", "Tmax": "tmax", "Tmin": "tmin"}[variable]
    m = folium.Map(location=[24.1, 88.0], zoom_start=7, tiles="OpenStreetMap", control_scale=True)
    add_map_layers(m, grid_df, col, variable)
    st_folium(m, use_container_width=True, height=580, returned_objects=[])

    # Recent history chart
    start = pd.Timestamp(obs_date) - pd.Timedelta(days=29)
    hist = ds.sel(time=slice(start, pd.Timestamp(obs_date)))
    if hist.sizes.get("time", 0) > 0:
        rows = []
        for var in ["rainfall", "tmax", "tmin"]:
            arr = hist[var].values
            for i, d in enumerate(pd.to_datetime(hist.time.values)):
                vals = arr[i][mask]
                rows.append({"Date": d, "Variable": var, "Value": float(np.nanmean(vals))})
        hist_df = pd.DataFrame(rows)
        hist_df["Variable"] = hist_df["Variable"].map({"rainfall": "Rainfall", "tmax": "Tmax", "tmin": "Tmin"})
        st.subheader("Recent observed conditions")
        fig = px.line(hist_df, x="Date", y="Value", color="Variable", markers=False)
        fig.update_layout(height=350, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

elif page == "Forecast & Spatial Map":
    st.subheader("Next-day forecast")
    st.caption(
        f"Target: {selected_target} · input window: "
        f"{pd.Timestamp(selected_target) - pd.Timedelta(days=7):%Y-%m-%d} to "
        f"{pd.Timestamp(selected_target) - pd.Timedelta(days=1):%Y-%m-%d}"
    )
    try:
        prediction, pred_grid = forecast_for_target_date(ds, mask, means, stds, model, selected_target)
    except Exception as exc:
        st.error(f"Forecast failed: {exc}")
        st.stop()

    regional = regional_means(pred_grid, mask)
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted rainfall", f"{regional['rainfall']:.2f} mm/day")
    c2.metric("Predicted Tmax", f"{regional['tmax']:.2f} °C")
    c3.metric("Predicted Tmin", f"{regional['tmin']:.2f} °C")

    variable = st.selectbox("Forecast map layer", ["Rainfall", "Tmax", "Tmin"], key="forecast_layer")
    grid_df = build_grid_table(ds, mask, pred_grid)
    col = {"Rainfall": "rainfall", "Tmax": "tmax", "Tmin": "tmin"}[variable]
    m = folium.Map(location=[24.1, 88.0], zoom_start=7, tiles="OpenStreetMap", control_scale=True)
    add_map_layers(m, grid_df, col, variable)
    st_folium(m, use_container_width=True, height=590, returned_objects=[])

    st.subheader("Grid-cell forecast values")
    st.dataframe(
        grid_df[["reference_name", "latitude", "longitude", "rainfall", "tmax", "tmin"]].round(2),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download forecast grid as CSV",
        grid_df.to_csv(index=False).encode("utf-8"),
        file_name=f"varuna_forecast_{selected_target}.csv",
        mime="text/csv",
    )
    if selected_target <= last_observed_date:
        actual = get_actual_grid(selected_target)
        if actual is not None:
            st.subheader("Forecast versus observation")
            actual_reg = regional_means(actual, mask)
            compare = pd.DataFrame({
                "Variable": ["Rainfall", "Tmax", "Tmin"],
                "Forecast": [regional["rainfall"], regional["tmax"], regional["tmin"]],
                "Observation": [actual_reg["rainfall"], actual_reg["tmax"], actual_reg["tmin"]],
            }).melt(id_vars="Variable", var_name="Series", value_name="Value")
            fig = px.bar(compare, x="Variable", y="Value", color="Series", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Historical target dates show a hindcast from the preceding seven days, alongside the recorded observation.")

elif page == "What-if Simulator":
    st.subheader("What-if simulator")
    st.write("Perturb the final observed day in the seven-day input window and compare the model response.")
    c1, c2, c3 = st.columns(3)
    with c1:
        rain_delta = st.slider("Rainfall change (mm)", -100.0, 100.0, 0.0, 5.0)
    with c2:
        tmax_delta = st.slider("Tmax change (°C)", -5.0, 5.0, 0.0, 0.5)
    with c3:
        tmin_delta = st.slider("Tmin change (°C)", -5.0, 5.0, 0.0, 0.5)

    if st.button("Run baseline vs scenario", type="primary"):
        from src.forecasting import run_scenario
        try:
            baseline, scenario, baseline_grid, scenario_grid = run_scenario(
                ds, mask, means, stds, model, selected_target,
                rain_delta, tmax_delta, tmin_delta
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
            rows.append({
                "Variable": f"{label} ({unit})",
                "Baseline": baseline[var],
                "Scenario": scenario[var],
                "Difference": scenario[var] - baseline[var],
            })
        comp = pd.DataFrame(rows)
        st.dataframe(comp.round(2), use_container_width=True, hide_index=True)
        chart = comp.melt(id_vars="Variable", value_vars=["Baseline", "Scenario"],
                          var_name="Run", value_name="Predicted value")
        st.plotly_chart(px.bar(chart, x="Variable", y="Predicted value", color="Run", barmode="group"),
                        use_container_width=True)

        map_variable = st.selectbox("Impact map variable", ["Rainfall", "Tmax", "Tmin"], key="impact_map")
        key = {"Rainfall": 0, "Tmax": 1, "Tmin": 2}[map_variable]
        delta_grid = scenario_grid[:, :, key] - baseline_grid[:, :, key]
        delta_table = build_grid_table(ds, mask, delta_grid[:, :, None], one_channel=True)
        delta_table["impact"] = delta_grid[mask]
        m = folium.Map(location=[24.1, 88.0], zoom_start=7, tiles="OpenStreetMap", control_scale=True)
        add_map_layers(m, delta_table, "impact", f"Scenario impact: {map_variable}")
        st_folium(m, use_container_width=True, height=560, returned_objects=[])
        st.caption("This is model sensitivity to altered inputs, not a physically validated causal simulation.")

elif page == "Model Performance":
    st.subheader("Model performance")
    st.write("The model was trained using a chronological split: training through 2022, validation in 2023–2024, and test data in 2025.")
    metrics_path = ROOT / "models" / "convlstm_attention_1day_metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.dataframe(metrics, use_container_width=True, hide_index=True)
        numeric = [c for c in metrics.columns if c.lower() in {"mae", "rmse", "r2", "mse"}]
        if numeric:
            st.bar_chart(metrics.set_index(metrics.columns[0])[numeric])
    else:
        st.info(
            "No `models/convlstm_attention_1day_metrics.csv` file was found. "
            "This page will not invent metrics. Export the real test-set metrics from your evaluation notebook "
            "to this path to display them here."
        )
    st.markdown("#### What the metrics mean")
    st.markdown("""
- **MAE**: average absolute forecast error, in the variable's original units.
- **RMSE**: gives larger errors more influence.
- **Forecast vs observation**: compare predictions with recorded values for dates where observations exist.
""")
    st.warning("The selected 2025 test period should remain the final evaluation period; avoid tuning model choices against it repeatedly.")

else:
    st.subheader("About VARUNA")
    st.markdown("""
**VARUNA** is an AI-assisted climate exploration prototype for West Bengal.

**Data currently used**
- Daily IMD rainfall, maximum temperature (Tmax), and minimum temperature (Tmin)
- Historical period in the prepared NetCDF file
- 14 selected grid cells with complete observations in the prepared study mask
- Rainfall and temperature represented on the same 7 × 5 grid

**Forecast model**
- ConvLSTM with temporal attention
- Seven historical days as input
- One-day prediction for each grid cell
- Forecast outputs are shown in physical units after inverse normalization

**Current limitations**
- This local prototype uses the prepared historical archive; it is not a live operational forecast service.
- Grid-cell centres are coarse model-grid locations, not weather-station coordinates.
- Place labels on the map are approximate reference labels for grid cells.
- Risk alerts, uncertainty intervals, and causal scenario claims are not yet validated operational products.
""")
