import streamlit as st
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import mplcursors  # Import mplcursors for hover functionality

# Path to your NetCDF file
NC_FILE_PATH = "climate-data.nc"

# Load dataset
@st.cache_data
def load_data():
    ds = xr.open_dataset(NC_FILE_PATH)
    return ds

ds = load_data()

# Extract latitude, longitude, and time information
lat = ds["lat"].values.flatten()
lon = ds["lon"].values.flatten()
time_values = ds["time"].values.astype(int)  # Convert to numpy array of integers

# Available variables to plot
variables = ["mai", "soy", "ri1", "ri2", "swh", "wwh"]

# Streamlit UI
st.title("🌍 Climate Data Dashboard")

# Sidebar Inputs
st.sidebar.header("🔍 Select Options")

# Use slider for time selection
time_index = st.sidebar.slider(
    "Select Time Index", 
    min_value=int(time_values.min()), 
    max_value=int(time_values.max()), 
    value=int(time_values[0]), 
    step=1
)

variable = st.sidebar.selectbox("Select Variable", variables)

# Adjust the time index selection for the dataset (assuming the time starts from 2015)
time_index_adjusted = time_index - 2015

# Extract the data for the selected variable and time index
data_slice = ds[variable].isel(time=time_index_adjusted)

# Extract latitudes and longitudes
lats = ds['lat'].values
lons = ds['lon'].values

# Set up the figure for the selected variable
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw={'projection': ccrs.PlateCarree()})

# Title for the plot
ax.set_title(f"{variable.upper()} Data for {time_index}")

# Add features to the plot
ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle=':')
ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.5)

# Plot the data for the selected variable
values = data_slice.values
mesh = ax.pcolormesh(lons, lats, values, transform=ccrs.PlateCarree(), cmap='viridis')

# Add colorbar
cbar = fig.colorbar(mesh, ax=ax, orientation='vertical', shrink=0.6)
cbar.set_label(f'{variable.upper()} Value')

# Add hover functionality
mplcursors.cursor(mesh, hover=True).connect("add", lambda sel: sel.annotation.set_text(
    f"Lat: {lats[sel.target.index[0]]:.2f}\nLon: {lons[sel.target.index[1]]:.2f}\nValue: {values[sel.target.index[0], sel.target.index[1]]:.2f}"
))

# Adjust layout
plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)

# Debugging Outputs
st.write(f"✅ Loaded data for **{variable}** at time index **{time_index}**.")
