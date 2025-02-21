import streamlit as st
import numpy as np
import joblib
import plotly.express as px
import pandas as pd

# 📂 Define crop variables
crop_variables = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
rf_models = {}

# 📌 Load Models
@st.cache_resource
def load_models():
    """Load trained crop yield prediction models."""
    models = {}
    for crop in crop_variables:
        try:
            model_path = f"models/{crop}_rf_model.pkl"
            models[crop] = joblib.load(model_path)
            st.sidebar.success(f"✅ {crop.upper()} model loaded")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading {crop} model: {e}")
    return models

rf_models = load_models()

# 📌 Load EOF Transformation
eof_patterns = None
try:
    eof_patterns = joblib.load("models/eof_solvers.pkl")  # Ensure correct filename
    st.write(eof_patterns[0])
    eof_patterns = np.array(eof_patterns)  # Convert to NumPy array if necessary
    st.sidebar.success("✅ EOF patterns loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Error loading EOF patterns: {e}")

# 📌 Sidebar UI
st.sidebar.title("Model Inputs")
crop_selection = st.sidebar.selectbox("Select Crop", crop_variables)

# 📌 Get Selected Model
selected_model = rf_models.get(crop_selection)

if selected_model:
    num_features = getattr(selected_model, "n_features_in_", None)

    # 📌 Display Expected Features
    if num_features:
        st.sidebar.info(f"Model expects {num_features} input features.")

    # 📌 User Input Sliders for 7 Climate Variables
    climate_vars = np.array([
        st.sidebar.slider("Precipitation (mm)", 0, 300, 100),
        st.sidebar.slider("Downward Longwave Radiation (W/m²)", 100, 400, 250),
        st.sidebar.slider("Downward Shortwave Radiation (W/m²)", 100, 1000, 500),
        st.sidebar.slider("Surface Wind Speed (m/s)", 0, 10, 5),
        st.sidebar.slider("Near-surface Air Temperature (°C)", -10, 40, 20),
        st.sidebar.slider("Daily Max Air Temperature (°C)", -10, 40, 30),
        st.sidebar.slider("Daily Min Air Temperature (°C)", -10, 40, 10)
    ]).reshape(1, -1)  # Shape: (1, 7)

    # 📌 Apply EOF Transformation
    eof_features = None
    st.write(eof_patterns.shape)
    if eof_patterns is not None:
        try:
            # Ensure EOF matrix dimensions match input shape
            if eof_patterns.shape[1] != climate_vars.shape[1]:
                st.sidebar.error(f"❌ EOF transformation shape mismatch! EOF expects {eof_patterns.shape[1]} features but got {climate_vars.shape[1]}.")
            else:
                eof_features = np.dot(climate_vars, eof_patterns.T)  # Shape: (1, num_EOF_features)

        except Exception as e:
            st.sidebar.error(f"❌ EOF Transformation Error: {e}")
            eof_features = None

    # 📌 Validate Feature Count
    if eof_features is None or (num_features and eof_features.shape[1] != num_features):
        st.sidebar.warning(f"⚠️ Model expects {num_features} features, but got {eof_features.shape[1] if eof_features is not None else 'None'}.")

    # 📌 Predict Yield
    if eof_features is not None and st.sidebar.button("Predict Yield"):
        try:
            predicted_yield = selected_model.predict(eof_features)
            st.sidebar.success(f"🌾 Predicted Yield: {predicted_yield[0]:.2f} tDM/ha")

            # 📌 Generate Latitude & Longitude Data for Visualization
            np.random.seed(42)
            num_points = 100
            df = pd.DataFrame({
                "Latitude": np.random.uniform(-50, 50, num_points),
                "Longitude": np.random.uniform(-180, 180, num_points),
                "tDM/ha": np.random.uniform(predicted_yield[0] * 0.8, predicted_yield[0] * 1.2, num_points)
            })

            # 📌 Create Heatmap Visualization
            fig = px.scatter_geo(
                df, 
                lat="Latitude", 
                lon="Longitude", 
                color="tDM/ha",
                projection="natural earth",
                title=f"{crop_selection.upper()} Predicted Yield",
                color_continuous_scale="viridis",
                hover_data=["tDM/ha"]
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.sidebar.error(f"❌ Prediction Error: {e}")

else:
    st.sidebar.warning("⚠️ No model loaded. Please check the model files.")

# 📌 App Description
st.markdown("**🌾 Crop Yield Prediction Model**")
st.write("This app predicts crop yields based on climate inputs using trained Random Forest models.")
print("EOF Pattern Shape:", eof_patterns.shape)
print("Climate Variable Shape:", climate_vars.shape)
