from fastapi import FastAPI
import xarray as xr
import os

app = FastAPI()

# Folder where NetCDF files are stored
DATA_FOLDER = "data"

# Function to load NetCDF data
def load_nc_data(file_name):
    file_path = os.path.join(DATA_FOLDER, file_name)
    if not os.path.exists(file_path):
        return None
    return xr.open_dataset(file_path)

@app.get("/datasets")
async def list_datasets():
    """Returns a list of available NetCDF files"""
    return {"datasets": os.listdir(DATA_FOLDER)}

@app.get("/data/{dataset}/{variable}/{time_index}")
async def get_nc_data(dataset: str, variable: str, time_index: int):
    """Fetches specific climate data"""
    ds = load_nc_data(dataset)
    if ds is None or variable not in ds:
        return {"error": "Dataset or variable not found"}
    
    # Replace NaN values with 0
    data = ds[variable].isel(time=time_index).fillna(0).values.tolist()
    lats = ds["lat"].values.tolist()
    lons = ds["lon"].values.tolist()
    
    return {"lats": lats, "lons": lons, "data": data}
