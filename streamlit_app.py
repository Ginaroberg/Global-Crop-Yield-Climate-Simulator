import streamlit as st
import numpy as np
import plotly.express as px
import pandas as pd
import requests
from scipy.ndimage import zoom
from plotly_resampler import FigureResampler

API_URL = "http://localhost:8000/data"

datasets = [
    "lpjml_picontrol_2015_2100.nc",
    "lpjml_ssp126_2015_2100.nc",
    "lpjml_ssp370_2015_2100.nc",
    "lpjml_ssp585_2015_2100.nc"
]

st.sidebar.title("Model Inputs")
dataset = st.sidebar.selectbox("Select Dataset", datasets)
variable = st.sidebar.selectbox("Select Variable", ["Maize", "Soy Beans", "First Season Rice","Second Season Rice", "Spring Wheat", "Winter Wheat"])
time_index = st.sidebar.slider("Select Time Index", 0, 100, 0)

crop_dict = {
    "Maize": "mai",
    "Soy Beans": "soy",
    "First Season Rice": "ri1",
    "Second Season Rice": "ri2",
    "Spring Wheat": "swh",
    "Winter Wheat": "wwh",
    }

var_converted = crop_dict[variable]

@st.cache_data(ttl=300)
def fetch_data(dataset, variable, time_index):
    try:
        response = requests.get(f"{API_URL}/{dataset}/{variable}/{time_index}?resolution=5")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

data = fetch_data(dataset, var_converted, time_index)

if "error" in data:
    st.error(f"❌ Error: {data['error']}")
else:
    st.success("✅ Data Loaded Successfully!")

    lats = np.array(data["lats"])
    lons = np.array(data["lons"])
    values = np.array(data["data"])

    def downsample_data(lats, lons, values, factor=0.5):
        return zoom(lats, factor), zoom(lons, factor), zoom(values, factor)

    lats, lons, values = downsample_data(lats, lons, values, factor=0.5)

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    df = pd.DataFrame({
        "Longitude": lon_grid.flatten(),
        "Latitude": lat_grid.flatten(),
        "tDM/ha": values.flatten() #Tons of Dry Matter per Hectacre
    })
    crop_dict = {
    "Maize": "mai",
    "Soy Beans": "soy",
    "First Season Rice": "ri1",
    "Second Season Rice": "ri2",
    "Spring Wheat": "swh",
    "Winter Wheat": "wwh",
    }

    year=2015+time_index

    fig = px.scatter_geo(
        df, 
        lat="Latitude", 
        lon="Longitude", 
        color="tDM/ha",
        projection="natural earth",
        title=f"{dataset} - {variable} Yields at Year {year}",
        color_continuous_scale="viridis",
        hover_data=["tDM/ha"]
    )

    fig.update_geos(
        showcountries=True,  
        countrycolor="black",
        countrywidth=5,

        showcoastlines=True,  
        coastlinecolor="gray",
        coastlinewidth=5,

        showland=True, landcolor="lightgray",
        showocean=True, oceancolor="white" 
    )

    fig = FigureResampler(fig)

    plot_area = st.empty()  # Placeholder for updates
    plot_area.plotly_chart(fig, use_container_width=True)
    st.markdown("The Crop Yields are measured in Tons of Dry matter Per Hectacre (tDM/ha)")
