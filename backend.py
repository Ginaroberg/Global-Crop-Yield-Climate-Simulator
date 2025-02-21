from fastapi import FastAPI
import xarray as xr
import os

app = FastAPI()

# 📂 NetCDF File Path
DATA_FOLDER = "data"
FILE_NAME = "outputs_piControl.nc"
file_path = os.path.join(DATA_FOLDER, FILE_NAME)

# 📌 Load NetCDF Data
def load_nc_data():
    if not os.path.exists(file_path):
        return None
    return xr.open_dataset(file_path)

dataset = load_nc_data()

@app.get("/data/{time_index}")
async def get_climate_data(time_index: int):
    """Fetches climate variables for all lat/lon points at a given time index"""
    if dataset is None:
        return {"error": "NetCDF file not found"}

    # Extract variables for the selected time
    climate_at_time = dataset.isel(time=time_index, member=0)

    # Extract specific variables
    tas = climate_at_time["tas"].values.tolist()  # Surface temperature
    pr = climate_at_time["pr"].values.tolist()  # Precipitation
    dtr = climate_at_time["diurnal_temperature_range"].values.tolist()  # Temp range

    lats = dataset["lat"].values.tolist()
    lons = dataset["lon"].values.tolist()

    return {"lats": lats, "lons": lons, "tas": tas, "pr": pr, "dtr": dtr}
