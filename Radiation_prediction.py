import streamlit as st
import numpy as np
import pandas as pd
import pickle
from scipy import stats
import datetime
import warnings
import os
import requests
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Solar Radiation Predictor",
    page_icon="☀️",
    layout="wide",
)

st.title("☀️ Solar Radiation Predictor")
st.markdown(
    "Adjust the sliders below — the prediction updates **instantly** as you move them."
)
st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "stacking_regressor_model.pkl")
    if not os.path.exists(model_path):
        # Download from external URL (replace with your actual direct download link)
        model_url = "https://drive.google.com/uc?export=download&id=1Vo-EjrAZrx1Svewo3oNLKX7CzUIO2pRe"
        response = requests.get(model_url)
        with open(model_path, "wb") as f:
            f.write(response.content)
    with open(model_path, "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
except FileNotFoundError:
    model = None
    st.error(
        "❌ Model file **stacking_regressor_model.pkl** not found.  \n"
        "Place it in the same folder as this app and restart."
    )
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing — 13 features (matches training exactly)
# Sunrise minute fixed at 15, Sunset minute fixed at 30 (dataset averages)
# ─────────────────────────────────────────────────────────────────────────────

BOXCOX_LAMBDA_PRESSURE  = -0.30
BOXCOX_LAMBDA_HUMIDITY  =  0.25

# Fixed internal defaults for hidden features
FIXED_RISE_MINUTE = 15
FIXED_SET_MINUTE  = 30

SCALER_MEANS = np.array([
    3.7636,   # Temperature (log+1)
    5.4731,   # Pressure    (boxcox)
    3.4841,   # Humidity    (boxcox)
    1.6094,   # Speed       (log+1)
    10.50,    # Month
    15.50,    # Day
    12.00,    # Hour
    29.50,    # Minute
    29.50,    # Second
     6.50,    # risehour
    15.00,    # riseminuter
    17.50,    # sethour
    45.00,    # setminute
])

SCALER_STDS = np.array([
    0.1062,   # Temperature
    0.0521,   # Pressure
    0.4318,   # Humidity
    0.5911,   # Speed
    1.1180,   # Month
    8.8034,   # Day
    3.4641,   # Hour
   17.3205,   # Minute
   17.3205,   # Second
    0.5774,   # risehour
    8.6603,   # riseminuter
    0.5774,   # sethour
   17.3205,   # setminute
])


def transform_and_scale(raw: dict) -> np.ndarray:
    temp     = np.log(raw["Temperature"] + 1)
    pressure = stats.boxcox(raw["Pressure"] + 1, BOXCOX_LAMBDA_PRESSURE)
    humidity = stats.boxcox(raw["Humidity"]  + 1, BOXCOX_LAMBDA_HUMIDITY)
    speed    = np.log(raw["Speed"] + 1)

    features = np.array([
        temp, pressure, humidity, speed,
        raw["Month"], raw["Day"], raw["Hour"],
        raw["Minute"], raw["Second"],
        raw["risehour"], raw["riseminuter"],
        raw["sethour"],  raw["setminute"],
    ], dtype=float)

    scaled = (features - SCALER_MEANS) / (SCALER_STDS + 1e-10)
    scaled = np.nan_to_num(scaled, nan=0.0)
    return scaled.reshape(1, -1)


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("🌡️ Weather Conditions")
    temperature = st.slider("Temperature (°F)",         min_value=20.0, max_value=100.0, value=60.0, step=0.5)
    pressure    = st.slider("Barometric Pressure (Hg)", min_value=25.0, max_value=32.0,  value=30.0, step=0.01)
    humidity    = st.slider("Humidity (%)",              min_value=1.0,  max_value=100.0, value=50.0, step=1.0)
    speed       = st.slider("Wind Speed (mph)",          min_value=0.0,  max_value=60.0,  value=10.0, step=0.5)

with col2:
    st.subheader("📅 Date")
    month = st.slider("Month",        min_value=1, max_value=12, value=10)
    day   = st.slider("Day of Month", min_value=1, max_value=31, value=15)

    st.subheader("🕐 Time of Day")
    time_input = st.time_input("Select time", value=datetime.time(12, 0, 0))
    hour   = time_input.hour
    minute = time_input.minute
    second = time_input.second

    st.subheader("🌅 Sunrise & Sunset")
    risehour = st.slider("Sunrise Hour", min_value=4,  max_value=8,  value=6)
    sethour  = st.slider("Sunset Hour",  min_value=16, max_value=20, value=18)

# ─────────────────────────────────────────────────────────────────────────────
# Instant prediction
# ─────────────────────────────────────────────────────────────────────────────
raw_input = {
    "Temperature":   temperature,
    "Pressure":      pressure,
    "Humidity":      humidity,
    "Speed":         speed,
    "Month":         month,
    "Day":           day,
    "Hour":          hour,
    "Minute":        minute,
    "Second":        second,
    "risehour":      risehour,
    "riseminuter":   FIXED_RISE_MINUTE,
    "sethour":       sethour,
    "setminute":     FIXED_SET_MINUTE,
}

X_input = transform_and_scale(raw_input)
if model is None:
    st.warning("No model loaded, cannot predict. Please place the pickle file and reload.")
    prediction = 0.0
else:
    prediction = float(model.predict(X_input)[0])
    prediction = max(0.0, prediction)


with col3:
    st.subheader("🔮 Live Prediction")
    st.markdown("<br>", unsafe_allow_html=True)

    st.metric(label="Predicted Solar Radiation", value=f"{prediction:.2f} W/m²")

    if prediction < 50:
        label, color = "🌑 Very Low / Nighttime", "#888888"
    elif prediction < 200:
        label, color = "🌤️ Low",                  "#f0a500"
    elif prediction < 500:
        label, color = "⛅ Moderate",             "#e6c200"
    elif prediction < 900:
        label, color = "🌞 High",                 "#f07800"
    else:
        label, color = "☀️ Very High / Peak",      "#e03000"

    st.markdown(
        f"<div style='font-size:1.3rem; font-weight:600; color:{color};'>{label}</div>",
        unsafe_allow_html=True,
    )
    st.progress(min(prediction / 1400.0, 1.0))
    st.divider()

    with st.expander("🔍 See transformed values sent to the model"):
        feature_names = [
            "Temperature (log+1)", "Pressure (BoxCox)", "Humidity (BoxCox)", "Speed (log+1)",
            "Month", "Day", "Hour", "Minute", "Second",
            "Sunrise Hour", "Sunrise Minute (fixed)", "Sunset Hour", "Sunset Minute (fixed)",
        ]
        df_show = pd.DataFrame({
            "Feature":              feature_names,
            "Raw value":            list(raw_input.values()),
            "Scaled (model input)": X_input.flatten().round(4),
        })
        st.dataframe(df_show, use_container_width=True, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: Stacking Regressor (Random Forest + Gradient Boosting + XGBoost + Ridge meta-learner)  ·  "
    "Dataset: HI-SEAS Weather Station Sep–Dec 2016  ·  Target: Solar Radiation (W/m²)")

