import streamlit as st
import numpy as np
import joblib
import xarray as xr
import plotly.express as px
import os
import requests
import gc  # Garbage collector for memory management
from sklearn.metrics import mean_squared_error  # For RMSE calculation
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut

# Initialize Geocoder
geolocator = Nominatim(user_agent="crop_yield_app")

# Function to Get Location Names from Lat/Lon
@st.cache_data  # Cache results to prevent repeated API calls
def get_location(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), exactly_one=True, language="en")
        if location:
            return location.address.split(", ")[-2:]  # Get city, country
        else:
            return "Unknown Location"
    except GeocoderTimedOut:
        return "Geocoder Timeout"



max_co2 = 9500.
max_ch4 = 0.8
max_so2 = 90.
max_bc = 9.

def normalize_inputs(data):
    return np.asarray(data) / np.asarray([max_co2, max_ch4, max_so2, max_bc])


def unnormalize_outputs(data):
    return np.asarray(data) * np.asarray([max_co2, max_ch4, max_so2, max_bc])


def global_mean(ds):
    weights = np.cos(np.deg2rad(ds.latitude))
    return ds.weighted(weights).mean(['latitude', 'longitude'])

def main():
    # Streamlit UI Setup
    st.title("Crop Yield Prediction from Emissions")
    st.sidebar.header("User Inputs: Greenhouse Gas Emissions")

    co2, ch4, so2, bc = emissions_ui()

    MODEL_DIR = "./models/"
    crop_mapping = {
        "Maize (mai)": "mai",
        "Rice Early Season (ri1)": "ri1",
        "Rice Late Season (ri2)": "ri2",
        "Soybean (soy)": "soy",
        "Spring Wheat (swh)": "swh",
        "Winter Wheat (wwh)": "wwh"
    }
    crop_models = {abbr: os.path.join(MODEL_DIR, f"{abbr}.pkl") for abbr in crop_mapping.values()}
    selected_crop_name = st.sidebar.selectbox("Select Crop Type", list(crop_mapping.keys()))
    selected_crop_abbr = crop_mapping[selected_crop_name]

    @st.cache_resource
    def load_crop_model(crop):
        model_path = crop_models.get(crop)
        if model_path and os.path.exists(model_path):
            st.sidebar.success(f"Model Loaded: {selected_crop_abbr}")
            return joblib.load(model_path)  # Using `joblib.load()`
        else:
            st.sidebar.error(f"Model not found for {crop}")
            return None
    
    model = load_crop_model(selected_crop_abbr)

    
    yields, uncertainty = crop_gp(model, selected_crop_abbr, co2, ch4, so2, bc)

    longitude, latitude = np.linspace(0, 360, 720), np.linspace(90, -90, 360)
    dataset = xr.DataArray(yields, coords={'latitude': (('latitude',), latitude), 'longitude': (('longitude',), longitude)})

    fig, ax = plt.subplots()
    dataset.plot.pcolormesh(ax=ax, cmap='coolwarm', vmax=6.)


    st.pyplot(fig)

    #Global Mean Yield
    global_mean_yield = np.nanmean(yields)
    st.write(f"**Global Mean Crop Yield:** {global_mean_yield:.2f} tDM/ha")

    #Max & Min Yield
    max_yield = np.nanmax(yields)
    st.write(f"**Maximum Predicted Yield:** {max_yield:.2f} tDM/ha")

    # Standard Deviation (Variability)
    std_dev_yield = np.nanstd(yields)
    st.write(f"**Standard Deviation of Yield:** {std_dev_yield:.2f} tDM/ha")

    # Finding Top 5 High-Yield Regions
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    flat_yields = yields.flatten()
    flat_lats = lat_grid.flatten()
    flat_lons = lon_grid.flatten()

    df_yield = pd.DataFrame({
        "Yield": flat_yields,
        "Latitude": flat_lats,
        "Longitude": flat_lons
    }).dropna()




    # Histogram of Yield Distribution
    st.write("Yield Distribution (Histogram)")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(yields.flatten(), bins=50, color="green", alpha=0.7, edgecolor="black")
    ax.set_title("Distribution of Predicted Crop Yields")
    ax.set_xlabel("Yield (tDM/ha)")
    ax.set_ylabel("Frequency")
    ax.grid(True)
    st.pyplot(fig)

    with st.expander("What Do These Units Mean?"):
        st.markdown("""
        ### **Crop Yield (tDM/ha)**
        - **1 tDM/ha = 1,000 kg of dry crop per hectare** (1 American Football field)
        
        **In terms of crops, 1 tDM/ha equals:**
        - **Rice**: 2,000 bowls of rice
        - **Maize**: 3,000 corn cobs 
        - **Wheat**: 1,800 loaves of bread

        ---

        ### **Greenhouse Gas Emissions**
        - **CO₂ (Carbon Dioxide)**: Measured in GtCO₂/year (Gigatons of CO₂ per year)
            - 1 GtCO₂ = Emissions from over 200,000 cars in a year
        
        - **CH₄ (Methane)**: Measured in GtCH₄/year (Gigatons of CH₄ per year)
            - 1 GtCH₄ = Equivalent to massive natural gas leaks

        - **SO₂ (Sulfur Dioxide)**: Measured in TgSO₂/year (Teragrams of SO₂ per year)
            - 1 TgSO₂ = Emissions from 150 coal power plants

        - **BC (Black Carbon) (Air Pollution Particles)**: Measured in TgBC/year (Teragrams of Black Carbon per year)
            - 1 TgBC = Air pollution from millions of diesel trucks

        """)




def emissions_ui():
    co2 = st.sidebar.slider("CO2 concentrations (GtCO2 / year)", 0.0, max_co2, 1800., 10., key='co2-slider')
    ch4 = st.sidebar.slider("Methane emissions (GtCH4 / year)", 0.0, max_ch4, 0.3, 0.005, key='ch4-slider')
    #  Just use global mean values for aerosol for simplicity
    so2 = st.sidebar.slider("SO2 emissions (TgSO2 / year)", 0.0, max_so2, 85., 1., key='so2-slider')
    bc = st.sidebar.slider("BC emissions (TgBC / year)", 0.0, max_bc, 7., 0.1, key='bc-slider')
    return normalize_inputs([co2, ch4, so2, bc])


# Load the Crop-Specific GP Model Using `joblib`


def crop_gp(model, selected_crop, co2, ch4, so2, bc):
    inputs = tf.convert_to_tensor([[co2, ch4, so2, bc]], dtype=tf.float64)
    posterior_mean, posterior_var = model.predict_y(inputs) # predicted mean of GP, predicted variance of GP
    posterior_stddev = np.sqrt(posterior_var)
    
    @st.cache_resource
    def load_mask(mask_path, crop):
        mask_file = os.path.join(mask_path, f"{crop}_mask.npy")
        return np.load(mask_file)
    
    mask_all_nan_by_col = load_mask("./mask/", selected_crop)
    
    posterior_yield_mean_full = np.full((1, 259200), np.nan)  # fill with NaN
    posterior_yield_mean_full[:, ~mask_all_nan_by_col] = posterior_mean

    posterior_yield_stddev_full = np.full((1, 259200), np.nan)  # fill with NaN
    posterior_yield_stddev_full[:, ~mask_all_nan_by_col] = posterior_stddev

    posterior_yield = np.reshape(posterior_yield_mean_full, [360, 720])
    posterior_yield_stddev = np.reshape(posterior_yield_stddev_full, [360, 720])
    
    return posterior_yield, posterior_yield_stddev

if __name__ == "__main__":
    main()
