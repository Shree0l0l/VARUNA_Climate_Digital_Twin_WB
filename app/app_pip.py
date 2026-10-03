
from pathlib import Path
import sys

import pandas as pd
import streamlit as st
import plotly.express as px


# Allow imports from the project root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.what_if import load_resources, run_what_if


st.set_page_config(
    page_title="VARUNA | What-if Simulator",
    page_icon="🌦️",
    layout="wide",
)

st.title("VARUNA — Climate What-if Simulator")
st.caption(
    "Explore how modified recent weather observations affect "
    "the trained model's next-day forecast."
)


@st.cache_resource
def get_resources():
    return load_resources()


try:
    ds, mask, means, stds, model = get_resources()
except Exception as exc:
    st.error(f"Could not load VARUNA resources: {exc}")
    st.info(
        "Check the NetCDF, normalization statistics, and trained "
        "attention model paths."
    )
    st.stop()


dates = pd.to_datetime(ds.time.values)
first_target_date = dates[7].date()
last_target_date = dates[-1].date()

st.subheader("1. Select the forecast date")

target_date = st.date_input(
    "Forecast target date",
    value=last_target_date,
    min_value=first_target_date,
    max_value=last_target_date,
    help=(
        "The selected date is the day to predict. "
        "VARUNA uses the seven days immediately before it."
    ),
)

st.caption(
    f"Study region: West Bengal | "
    f"Selected grid cells: {int(mask.sum())} | "
    f"Available data: {dates[0].date()} to {dates[-1].date()}"
)

st.subheader("2. Modify the final observed day")

st.write(
    "Changes apply to the last day in the seven-day input window, "
    "at every selected grid cell. The other six days remain unchanged."
)

col1, col2, col3 = st.columns(3)

with col1:
    rainfall_change = st.slider(
        "Rainfall change (mm)",
        min_value=-100.0,
        max_value=100.0,
        value=0.0,
        step=5.0,
        help="Negative changes cannot reduce rainfall below zero.",
    )

with col2:
    tmax_change = st.slider(
        "Tmax change (°C)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
    )

with col3:
    tmin_change = st.slider(
        "Tmin change (°C)",
        min_value=-5.0,
        max_value=5.0,
        value=0.0,
        step=0.5,
    )

st.info(
    "These are experimental input perturbations, not official "
    "weather forecasts or physically validated climate scenarios."
)

if st.button("Run simulation", type="primary"):
    with st.spinner("Running baseline and scenario predictions..."):
        try:
            comparison, baseline, scenario, history_dates = run_what_if(
                ds=ds,
                mask=mask,
                means=means,
                stds=stds,
                model=model,
                target_date=target_date,
                rainfall_change=rainfall_change,
                tmax_change=tmax_change,
                tmin_change=tmin_change,
            )
        except Exception as exc:
            st.error(f"Simulation failed: {exc}")
            st.stop()

    st.subheader("3. Forecast comparison")

    st.write(
        f"**Forecast date:** {target_date}  \n"
        f"**Input window:** {history_dates[0].date()} to "
        f"{history_dates[-1].date()}"
    )

    display_df = comparison.copy()
    for col in ["Baseline", "Scenario", "Difference"]:
        display_df[col] = display_df[col].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("4. Baseline versus scenario")

    chart_df = comparison.melt(
        id_vars="Variable",
        value_vars=["Baseline", "Scenario"],
        var_name="Forecast",
        value_name="Predicted value",
    )

    fig = px.bar(
        chart_df,
        x="Variable",
        y="Predicted value",
        color="Forecast",
        barmode="group",
        title="Regional mean next-day forecast",
        text_auto=".2f",
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("5. Interpretation")

    for _, row in comparison.iterrows():
        difference = row["Difference"]
        st.write(
            f"**{row['Variable']}** changed by "
            f"{difference:+.2f} relative to the baseline."
        )

    st.caption(
        "Regional means summarize the selected grid cells. "
        "They do not represent every district or weather station. "
        "The model's response is a sensitivity result, not proof "
        "of physical causation."
    )
else:
    st.write(
        "Adjust the sliders and click **Run simulation** to compare "
        "the baseline forecast with the modified-input forecast."
    )