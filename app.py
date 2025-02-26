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

max_co2 = 9500.
max_ch4 = 0.8
max_so2 = 90.
max_bc = 9.

def normalize_inputs(data):
    return np.asarray(data) / np.asarray([max_co2, max_ch4, max_so2, max_so2])


def unnormalize_outputs(data):
    return np.asarray(data) * np.asarray([max_co2, max_ch4, max_so2, max_so2])


def global_mean(ds):
    weights = np.cos(np.deg2rad(ds.latitude))
    return ds.weighted(weights).mean(['latitude', 'longitude'])

def main():
    # Streamlit UI Setup
    st.title("Crop Yield Prediction from Emissions")
    st.sidebar.header("User Inputs: Greenhouse Gas Emissions")

    co2, ch4, so2, bc = emissions_ui()

    MODEL_DIR = "./models/"
    crop_variables = ['mai', 'ri1', 'ri2', 'soy', 'swh', 'wwh']
    crop_models = {crop: os.path.join(MODEL_DIR, f"{crop}.pkl") for crop in crop_variables}
    selected_crop = st.sidebar.selectbox("Select Crop Type", crop_variables)

    @st.cache_resource
    def load_crop_model(crop):
        model_path = crop_models.get(crop)
        print(model_path)
        print(os.path.exists(model_path))
        if model_path and os.path.exists(model_path):
            st.sidebar.success(f"✅ Model Loaded: {selected_crop}")
            return joblib.load(model_path)  # ✅ Using `joblib.load()`
        else:
            st.sidebar.error(f"❌ Model not found for {crop}")
            return None
    
    model = load_crop_model(selected_crop)

    
    yields, uncertainty = crop_gp(model, co2, ch4, so2, bc)

    longitude, latitude = np.linspace(0, 360, 720), np.linspace(90, -90, 360)
    dataset = xr.DataArray(yields, coords={'latitude': (('latitude',), latitude), 'longitude': (('longitude',), longitude)})

    fig, ax = plt.subplots()
    dataset.plot(ax=ax, cmap='coolwarm', vmax=6.)


    st.pyplot(fig)
    
    


def emissions_ui():
    co2 = st.sidebar.slider("CO2 concentrations (GtCO2)", 0.0, max_co2, 1800., 10., key='co2-slider')
    ch4 = st.sidebar.slider("Methane emissions (GtCH4 / year)", 0.0, max_ch4, 0.3, 0.005, key='ch4-slider')
    #  Just use global mean values for aerosol for simplicity
    so2 = st.sidebar.slider("SO2 emissions (TgSO2 / year)", 0.0, max_so2, 85., 1., key='so2-slider')
    bc = st.sidebar.slider("BC emissions (TgBC / year)", 0.0, max_bc, 7., 0.1, key='bc-slider')
    return normalize_inputs([co2, ch4, so2, bc])


# Load the Crop-Specific GP Model Using `joblib`


def crop_gp(model, co2, ch4, so2, bc):
    inputs = tf.convert_to_tensor([[co2, ch4, so2, bc]], dtype=tf.float64)
    posterior_mean, posterior_var = model.predict_y(inputs) # predicted mean of GP, predicted variance of GP
    posterior_stddev = np.sqrt(posterior_var)
    
    mask_all_nan_by_col = np.load('./data/mask.npy')
    st.write(mask_all_nan_by_col.shape)
    
    posterior_yield_mean_full = np.full((1, 259200), np.nan)  # fill with NaN
    posterior_yield_mean_full[:, ~mask_all_nan_by_col] = posterior_mean

    posterior_yield_stddev_full = np.full((1, 259200), np.nan)  # fill with NaN
    posterior_yield_stddev_full[:, ~mask_all_nan_by_col] = posterior_stddev

    posterior_yield = np.reshape(posterior_yield_mean_full, [360, 720])
    posterior_yield_stddev = np.reshape(posterior_yield_stddev_full, [360, 720])
    
    return posterior_yield, posterior_yield_stddev

if __name__ == "__main__":
    main()
