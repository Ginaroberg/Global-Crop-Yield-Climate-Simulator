import streamlit as st
import numpy as np
import joblib
import xarray as xr
import plotly.express as px
import os
from sklearn.metrics import mean_squared_error  # For RMSE calculation
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go



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

def main():
    # Initializations
    if "selected_country" not in st.session_state:
        st.session_state.selected_country = None
    
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
    # Load model
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

    
    # Get predictions
    yields, uncertainty = crop_gp(model, selected_crop_abbr, co2, ch4, so2, bc)

    # Choroplth Map of Predicted Crop Yields
    longitude, latitude = np.linspace(-179.75, 179.75, 720), np.linspace(89.75, -89.75, 360)
    dataset = xr.DataArray(yields, coords={'latitude': (('latitude',), latitude), 'longitude': (('longitude',), longitude)})
    
    # Create a Figure
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=dataset.values,
        x=longitude,
        y=latitude,
        colorbar=dict(
            title=dict(text="Crop Yield (tDM/ha)", side="right"),
            thickness=10,
            len=1,
            x=1.02
            ),
        hoverinfo="skip",
        colorscale="Viridis"  # Use the Viridis colormap
    ))
    fig.update_layout(
        title='Crop Yield Prediction from Emissions',
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=500,
        width=1500,
        xaxis=dict(
            fixedrange=True,
            showgrid=False,
            tickvals=list(range(-180, 181, 30)),
            ticktext=[str(i) + "°" for i in range(-180, 181, 30)]
        ),
        yaxis=dict(fixedrange=True,
                   showgrid=False,
                   zeroline=False,
                   tickvals=list(range(-90, 91, 30)),
                   ticktext=[str(i) + "°" for i in range(-90, 91, 30)]),
        plot_bgcolor="white"
    )
    st.plotly_chart(fig, use_container_width=True)
    

    #Global Mean Yield
    global_mean_yield = np.nanmean(yields)
    st.write(f"**Global Mean Crop Yield:** {global_mean_yield:.2f} tDM/ha")

    #Max & Min Yield
    max_yield = np.nanmax(yields)
    st.write(f"**Maximum Predicted Yield:** {max_yield:.2f} tDM/ha")

    # Standard Deviation (Variability)
    std_dev_yield = np.nanstd(yields)
    st.write(f"**Standard Deviation of Yield:** {std_dev_yield:.2f} tDM/ha")



    # Tnteractive choropleth map showing crop yield sum by country
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    df = pd.DataFrame({
    "lat": lat_grid.ravel(),
    "lon": lon_grid.ravel(),
    "yield": yields.ravel()
    })
    country_boundaries = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).drop(columns=["pop_est", "gdp_md_est"])
    yield_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    yield_countries = gpd.sjoin(yield_gdf, country_boundaries, how="left", predicate="within")
    country_yield = yield_countries.groupby("name")["yield"].sum().reset_index()

    fig = px.choropleth(
        country_yield,
        locations="name",
        locationmode="country names",
        color="yield",
        color_continuous_scale="Viridis",
        projection='orthographic',
        
    )
    fig.update_layout(height=700, width=700)
    fig.update_coloraxes(colorbar_title="Crop Yield (tDM/ha)",
                         colorbar_thickness=10,
                         colorbar_len=0.75,
                         )
    st.plotly_chart(fig)
    



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


if __name__ == "__main__":
    main()
