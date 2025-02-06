import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import requests

# FastAPI backend URL
API_URL = "http://localhost:8000/data"

# Available datasets
datasets = [
    "lpjml_picontrol_2015_2100.nc",
    "lpjml_ssp126_2015_2100.nc",
    "lpjml_ssp370_2015_2100.nc",
    "lpjml_ssp585_2015_2100.nc"
]

# Streamlit Inputs
dataset = st.sidebar.selectbox("Select Dataset", datasets)
variable = st.sidebar.selectbox("Select Variable", ["mai", "soy", "ri1", "ri2", "swh", "wwh"])
time_index = st.sidebar.slider("Select Time Index", min_value=0, max_value=100, value=0)

# Fetch data from FastAPI
response = requests.get(f"{API_URL}/{dataset}/{variable}/{time_index}")
data = response.json()

if 'error' in data:
    st.error(data['error'])
else:
    # Extract the data
    lats = np.array(data["lats"])
    lons = np.array(data["lons"])
    values = np.array(data["data"])

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_title(f"{dataset} for {variable.upper()} Data for Time Index {time_index}")
    
    # Set the land and ocean features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linestyle=':')
    ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)
    ax.add_feature(cfeature.OCEAN, color='white')  # Set ocean color to white
    
    # Plot the data using pcolormesh
    mesh = ax.pcolormesh(lons, lats, values, transform=ccrs.PlateCarree(), cmap='viridis')

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.6)
    cbar.set_label(f'{variable.upper()} Value')

    # Display the plot
    st.pyplot(fig)
