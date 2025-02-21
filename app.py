import streamlit as st
import numpy as np
import joblib
import xarray as xr
import plotly.express as px
import os
import requests
import gc  # Garbage collector for memory management
from sklearn.metrics import mean_squared_error  # For RMSE calculation

# Streamlit UI Setup
st.title("Crop Yield Prediction from Emissions")
st.sidebar.header("User Inputs: Greenhouse Gas Emissions")

# User Inputs for Emissions
co2 = st.sidebar.slider("CO₂ Concentration (ppm)", 200, 600, 400)
ch4 = st.sidebar.slider("CH₄ Concentration (ppb)", 1000, 2500, 1800)
so2 = st.sidebar.slider("SO₂ Emissions (kt)", 0, 100, 50)
bc  = st.sidebar.slider("Black Carbon Emissions (kt)", 0, 100, 20)

# Path to the folder containing the model files
MODEL_DIR = "GP_emission_to_crop_yield_pkl"
crop_variables = ['mai', 'ri1', 'ri2', 'soy', 'swh', '2wwh']
crop_models = {crop: os.path.join(MODEL_DIR, f"GP_emi_to_{crop}.pkl") for crop in crop_variables}

# Select Crop Type
selected_crop = st.sidebar.selectbox("Select Crop Type", crop_variables)

# Convert Single BC and SO₂ Values into Multiple Components
bc_values = np.full(5, bc / 5)   # Spread BC across BC_0 to BC_4
so2_values = np.full(5, so2 / 5) # Spread SO₂ across SO₂_0 to SO₂_4

# Load the Crop-Specific GP Model Using `joblib`
@st.cache_resource
def load_crop_model(crop):
    model_path = crop_models.get(crop)
    if model_path and os.path.exists(model_path):
        return joblib.load(model_path)  # ✅ Using `joblib.load()`
    else:
        st.sidebar.error(f"❌ Model not found for {crop}")
        return None

model = load_crop_model(selected_crop)

# Load Lat/Lon Data from FastAPI
FASTAPI_URL = "http://localhost:8000"
if st.sidebar.button("Load Locations"):
    try:
        response = requests.get(f"{FASTAPI_URL}/data/0")  # Using time index 0 to fetch lat/lon
        if response.status_code == 200:
            data = response.json()
            lats = np.array(data["lats"])
            lons = np.array(data["lons"])

            # # Limit number of locations to prevent crashes
            # MAX_LOCATIONS = 5000
            # if len(lats) * len(lons) > MAX_LOCATIONS:
            #     st.warning(f"⚠️ Too many locations! Reducing to {MAX_LOCATIONS} for performance.")
            #     lats = lats[:int(np.sqrt(MAX_LOCATIONS))]
            #     lons = lons[:int(np.sqrt(MAX_LOCATIONS))]

            st.sidebar.success("✅ Locations Loaded")
        else:
            st.sidebar.error("Failed to fetch location data")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Predict Crop Yields Using Batching to Prevent Crashes
if model and "lats" in locals() and "lons" in locals():
    st.sidebar.success(f"✅ Model Loaded: {selected_crop}")

    # Prepare Emission Inputs for Each Location (Repeat for Spatial Grid)
    num_locations = len(lats) * len(lons)
    emission_inputs = np.tile(
        np.concatenate(([co2, ch4], bc_values, so2_values)), (num_locations, 1)
    )  # Shape: (num_points, 12)

    # # Ensure Input Matches Model Expectations
    # expected_features = model.data[0].shape[1] 
    # if emission_inputs.shape[1] < expected_features:
    #     missing_features = expected_features - emission_inputs.shape[1]
    #     padding = np.zeros((emission_inputs.shape[0], missing_features))  # Add zero padding
    #     emission_inputs = np.hstack([emission_inputs, padding])  # Ensure correct input size

    # Batch Processing for Large Data
    batch_size = 1000  # Adjust for available memory
    num_batches = int(np.ceil(emission_inputs.shape[0] / batch_size))

    pred_means = []
    pred_stds = []

    for i in range(num_batches):
        batch = emission_inputs[i * batch_size : (i + 1) * batch_size]
        posterior_mean, posterior_var = model.predict_y(batch)
        
        # Append only the required amount
        pred_means.append(posterior_mean.numpy()[:batch.shape[0]])
        pred_stds.append(np.sqrt(posterior_var.numpy()[:batch.shape[0]]))

    # Merge results
    posterior_mean = np.concatenate(pred_means).flatten()
    posterior_stddev = np.concatenate(pred_stds).flatten()

    # Fix Reshape Error by Checking Array Sizes
    num_predictions = posterior_mean.size

    if num_predictions > num_locations:
        st.warning(f"⚠️ Model produced more predictions ({num_predictions}) than expected ({num_locations}). Truncating...")
        posterior_mean = posterior_mean[:num_locations]
        posterior_stddev = posterior_stddev[:num_locations]
    elif num_predictions < num_locations:
        st.error(f"❌ Model output size ({num_predictions}) is smaller than expected grid size ({num_locations}).")
        st.stop()  # Prevent further execution if this happens

    # Reshape Predictions to (lat, lon)
    predicted_yield = posterior_mean.reshape(len(lats), len(lons))
    predicted_stddev = posterior_stddev.reshape(len(lats), len(lons))

    # Calculate RMSE
    # true_yields = np.random.uniform(low=predicted_yield.min(), high=predicted_yield.max(), size=predicted_yield.shape)  # Simulated true values for RMSE
    # rmse = np.sqrt(mean_squared_error(true_yields.flatten(), predicted_yield.flatten()))
    # st.write(f"📊 **RMSE (Root Mean Squared Error):** {rmse:.4f}")

    # Free Memory After Prediction
    del model
    gc.collect()

    # 📌 🎨 Plot Predicted Crop Yield
    st.write(f"### Predicted Crop Yield for {selected_crop}")
    fig = px.imshow(predicted_yield, 
                    origin="lower", 
                    color_continuous_scale="viridis",
                    labels=dict(color="Yield (tDM/ha)"))
    fig.update_layout(title=f"Predicted Yield ({selected_crop}, Adjusted for Emissions)")
    st.plotly_chart(fig, use_container_width=True)

    # 📌 🎨 Plot Standard Deviation (Uncertainty)
    st.write(f"### Prediction Uncertainty for {selected_crop}")
    fig_std = px.imshow(predicted_stddev, 
                        origin="lower", 
                        color_continuous_scale="plasma",
                        labels=dict(color="StdDev (tDM/ha)"))
    fig_std.update_layout(title=f"Prediction Uncertainty ({selected_crop})")
    st.plotly_chart(fig_std, use_container_width=True)
