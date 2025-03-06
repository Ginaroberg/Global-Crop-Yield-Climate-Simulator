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

from shapely.geometry import Point

#Load Natural Earth Country Boundaries
@st.cache_data
def load_country_shapefile():
    shapefile_path = "110m_cultural/ne_110m_admin_0_countries.shp" 
    return gpd.read_file(shapefile_path)

gdf_countries = load_country_shapefile()

#Aggregate Crop Yield by Country
def get_country_yield(latitude, longitude, yields):
    """
    Aggregates crop yield by country using Natural Earth country boundaries.
    https://www.naturalearthdata.com/downloads/110m-cultural-vectors/
    """
    lat_grid, lon_grid = np.meshgrid(latitude, longitude, indexing="ij")
    df = pd.DataFrame({
        "lat": lat_grid.ravel(),
        "lon": lon_grid.ravel(),
        "yield": yields.ravel()
    })

    yield_gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")

    joined = gpd.sjoin(yield_gdf, gdf_countries, how="left", predicate="within")

    country_yield = joined.groupby("NAME")["yield"].sum().reset_index()

    return country_yield


def main():
    st.set_page_config(layout="wide")
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
        # **🌍 Create Side-by-Side Layout (Heatmap & Statistics Table)**
    col1, col2 = st.columns([3, 1])  # Adjust width ratios for better layout

    # **📊 Heatmap in Left Column**
    with col1:
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
            xaxis_title="Longitude",
            yaxis_title="Latitude",
            height=500,
            width=1200,
            xaxis=dict(
                fixedrange=True,
                showgrid=False,
                tickvals=list(range(-180, 181, 30)),
                ticktext=[str(i) + "°" for i in range(-180, 181, 30)]
            ),
            yaxis=dict(
                fixedrange=True,
                showgrid=False,
                zeroline=False,
                tickvals=list(range(-90, 91, 30)),
                ticktext=[str(i) + "°" for i in range(-90, 91, 30)]
            ),
            plot_bgcolor="white"
        )

        st.plotly_chart(fig, use_container_width=True)

    # **Simple Yield Statistics Table in Right Column**
    with col2:
        global_mean_yield = np.nanmean(yields)
        max_yield = np.nanmax(yields)
        std_dev_yield = np.nanstd(yields)
        for _ in range(7): # spacing
            st.write("")

        st.write(f"**Global Mean Yield:** {global_mean_yield:.2f} tDM/ha")
        st.write(f"**Maximum Yield:** {max_yield:.2f} tDM/ha")
        st.write(f"**Standard Deviation:** {std_dev_yield:.2f} tDM/ha")


    # Interactive chloropleth and bar chart
    country_yield = get_country_yield(latitude, longitude, yields)
   
    st.title("Top 20 Countries with Highest Crop Yield")
    top_20_countries = country_yield.nlargest(20, "yield").sort_values("yield", ascending=True).copy()

    col1, col2 = st.columns([1.2, 1.8]) 

    #Bar Chart in Left Column**
    with col1:

        selected_country = st.selectbox(
                "Select a country to highlight:", 
                ["None"] + list(top_20_countries["NAME"])
            )
        bar_colors = ["red" if country == selected_country else "blue" for country in top_20_countries["NAME"]]

        bar_fig = go.Figure()
        bar_fig.add_trace(go.Bar(
            x=top_20_countries["yield"],
            y=top_20_countries["NAME"],
            orientation="h",
            marker=dict(color=bar_colors),
            text=top_20_countries["yield"].round(2), 
            textposition="outside", 
            textfont=dict(size=14), 
        ))

        bar_fig.update_layout(
 
            xaxis_title="Total Yield (tDM/ha)",
            yaxis_title="Country",
            height=600,
            showlegend=False,
            xaxis=dict(tickfont=dict(size=12)), 
            yaxis=dict(tickfont=dict(size=14)), 
        )

        st.plotly_chart(bar_fig, use_container_width=True)

    #Choropleth Map in Right Column**
    with col2:

        def plot_choropleth(selected_country):
            fig = px.choropleth(
                country_yield,
                locations="NAME",
                locationmode="country names",
                color="yield",
                color_continuous_scale="Viridis",
                projection='orthographic',
            )

            if selected_country and selected_country != "None":
                highlight_country = gdf_countries[gdf_countries["NAME"] == selected_country]

                if not highlight_country.empty:
                    country_center = highlight_country.geometry.centroid.iloc[0]

                    fig.update_layout(
                        geo=dict(
                            projection_rotation=dict(lon=country_center.x, lat=country_center.y),
                            projection_type="orthographic"
                        )
                    )

                    boundary = highlight_country.geometry.boundary.iloc[0]

                    lon_vals, lat_vals = [], []

                    if boundary.geom_type == "MultiLineString":
                        for line in boundary.geoms: 
                            lon_vals.extend(list(line.xy[0])) 
                            lat_vals.extend(list(line.xy[1])) 
                    elif boundary.geom_type == "LineString":
                        lon_vals = list(boundary.xy[0])
                        lat_vals = list(boundary.xy[1]) 

                    fig.add_trace(go.Scattergeo(
                        lon=lon_vals,
                        lat=lat_vals,
                        mode="lines",
                        line=dict(width=3, color="red"),
                        name=f"Outline: {selected_country}"
                    ))

            fig.update_layout(
                height=700, width=900,
                margin=dict(l=50, r=50, t=50, b=50)
            )
            fig.update_coloraxes(
                colorbar_title="Crop Yield (tDM/ha)",
                colorbar_thickness=12,
                colorbar_len=0.75
            )

            return fig

        st.plotly_chart(plot_choropleth(selected_country), use_container_width=True)

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
