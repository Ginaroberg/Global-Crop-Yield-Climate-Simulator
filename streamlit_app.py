import streamlit as st
import numpy as np
import joblib
import os
import requests
import gc  # Garbage collector for memory management
import plotly.express as px
from sklearn.metrics import mean_squared_error  # For RMSE calculation
import xarray as xr
import plotly.graph_objects as go



# 📌 Streamlit UI Setup
st.title("Crop Yield Prediction from Emissions (Random Forest)")
st.sidebar.header("User Inputs: Greenhouse Gas Emissions")

# 📌 User Inputs for Emissions
co2 = st.sidebar.slider(" Concentration (ppm)", 200, 600, 400)
ch4 = st.sidebar.slider("Methane emissions (CH₄) Concentration (ppb)", 1000, 2500, 1800)
so2 = st.sidebar.slider("SO₂ Emissions (kt)", 0, 100, 50)
bc  = st.sidebar.slider("Black Carbon Emissions (kt)", 0, 100, 20)

# 📌 Model Paths
MODEL_DIR = "models"
crop_variables = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
crop_models = {crop: os.path.join(MODEL_DIR, f"{crop}_rf_model.pkl") for crop in crop_variables}

# 📌 Select Crop Type
selected_crop = st.sidebar.selectbox("Select Crop Type", crop_variables)

# Convert Single BC and SO₂ Values into Multiple Components
bc_values = np.full(5, bc / 5)   # Spread BC across BC_0 to BC_4
so2_values = np.full(5, so2 / 5) # Spread SO₂ across SO₂_0 to SO₂_4

# 📌 Load Random Forest Model
@st.cache_resource
def load_crop_model(crop):
    model_path = crop_models.get(crop)
    if model_path and os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        st.sidebar.error(f"❌ Model not found for {crop}")
        return None

model = load_crop_model(selected_crop)

# 📌 Load Lat/Lon Data from FastAPI
FASTAPI_URL = "http://localhost:8000"
if st.sidebar.button("Load Locations"):
    try:
        response = requests.get(f"{FASTAPI_URL}/data/0")  # Fetch lat/lon from index 0
        if response.status_code == 200:
            data = response.json()
            lats = np.array(data["lats"])
            lons = np.array(data["lons"])
            st.sidebar.success("✅ Locations Loaded")
        else:
            st.sidebar.error("Failed to fetch location data")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# 📌 Predict Crop Yields Using Batching to Prevent Memory Issues
if model and "lats" in locals() and "lons" in locals():
    st.sidebar.success(f"✅ Model Loaded: {selected_crop}")

    num_locations = len(lats) * len(lons)
    
    # Prepare Inputs for Prediction
    emission_inputs = np.tile(
        np.concatenate(([co2, ch4], bc_values, so2_values)), (num_locations, 1)
    ) 

    # Batch Processing
    batch_size = 1000  
    num_batches = int(np.ceil(emission_inputs.shape[0] / batch_size))
    
    pred_means = []
    for i in range(num_batches):
        batch = emission_inputs[i * batch_size : (i + 1) * batch_size]
        predictions = model.predict(batch)

        # Convert to NumPy array if needed
        if isinstance(predictions, tuple):
            predictions = predictions[0]  
        if isinstance(predictions, xr.DataArray):
            predictions = predictions.values  
        
        pred_means.append(predictions.flatten())

    # Merge Predictions
    predicted_yield = np.concatenate(pred_means).flatten()

    # Fix Shape Issues
    if predicted_yield.size > num_locations:
        predicted_yield = predicted_yield[:num_locations]
    elif predicted_yield.size < num_locations:
        st.error(f"❌ Model output size ({predicted_yield.size}) does not match expected ({num_locations}).")
        st.stop()

    # Reshape Predictions
    predicted_yield = predicted_yield.reshape(len(lats), len(lons))

    # Free Memory
    del model
    gc.collect()

    # 📌 Plot using `scattergeo`
    fig = go.Figure()

    fig.add_trace(go.Scattergeo(
        lon=lons.repeat(len(lats)), 
        lat=np.tile(lats, len(lons)),
        text=[f"Yield: {y:.2f} tDM/ha" for y in predicted_yield.flatten()],  # Tooltip
        marker=dict(
            size=6,  # Marker size (adjustable)
            color=predicted_yield.flatten(),  # Color by predicted yield
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="Yield (tDM/ha)"),
            opacity=0.8
        ),
        mode="markers"
    ))

    # Configure World Map Layout
    fig.update_layout(
        title=f"Global Crop Yield Predictions ({selected_crop})",
        geo=dict(
            projection_type="equirectangular",  # Better world projection
            showland=True,
            showcoastlines=True,
            showcountries=True,
            landcolor="rgb(217, 217, 217)",
            coastlinecolor="rgb(255, 255, 255)",
        )
    )

    # 📌 🎨 Show Map in Streamlit
    st.plotly_chart(fig, use_container_width=True)