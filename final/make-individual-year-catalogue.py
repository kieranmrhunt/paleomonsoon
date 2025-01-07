import matplotlib.pyplot as plt
from cartopy import crs as ccrs
import geopandas as gpd
import numpy as np
import colormaps as cmaps
import os
from functions import *
import matplotlib.pyplot as plt
from cartopy.feature import ShapelyFeature
from matplotlib.colors import LinearSegmentedColormap
import xarray as xr


# Load the boundary shapefile for India
df = gpd.read_file("/home/users/rz908899/geodata/mapindia/India_Boundary.shp")

# Define the colormap and geographic bounds
cmap = cmaps.BlueWhiteOrangeRed_r.discrete(17)
lons = np.arange(66.5, 100.25, 0.25)
lats = np.arange(6.5, 38.75, 0.25)

# Fetch CNN ensemble mean precipitation data
cnn_ensemble_prcp, lons, lats, cnn_year_list = fetch_cnn_ensemble('IMD', return_all_models = True)
cnn_ensemble_mean_prcp = np.mean(cnn_ensemble_prcp, axis=0)  # Average across models
year_mask = (cnn_year_list >= 1501) & (cnn_year_list <= 1994)
cnn_ensemble_mean_prcp = cnn_ensemble_mean_prcp[year_mask]
cnn_year_list = cnn_year_list[year_mask]

cnn_ensemble_prcp = cnn_ensemble_prcp[:, year_mask, :, :]

# Create a directory for the output files if it doesn't already exist
output_dir = "full-individual-years"
os.makedirs(output_dir, exist_ok=True)


print(np.shape(cnn_ensemble_prcp))
print(np.shape(cnn_ensemble_mean_prcp))

# Create an xarray dataset
ds = xr.Dataset(
    {
        "ensemble_mean": (["time", "lat", "lon"], cnn_ensemble_mean_prcp),
        "ensemble_members": (["member", "time", "lat", "lon"], cnn_ensemble_prcp),
    },
    coords={
        "lon": lons,
        "lat": lats,
        "time": cnn_year_list,
        "member": np.arange(cnn_ensemble_prcp.shape[0]), 
    },
)

# Save the dataset to a NetCDF file
ds.to_netcdf("../data-for-zenodo/cnn_ensemble_prcp.nc")

# Loop over each year in the filtered list
for year, prcp in zip(cnn_year_list, cnn_ensemble_mean_prcp):
    print(year)
    # Create a figure with a PlateCarree projection
    fig, ax = plt.subplots(figsize=(6, 5), subplot_kw={'projection': ccrs.PlateCarree()})

    # Plot the precipitation data
    c = ax.pcolormesh(lons, lats, prcp, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
    df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.5)
    ax.set_ylim([6.5, 38.5])
    ax.set_xlim([66.5, 100])
    ax.set_title(f"{year} CNN ensemble mean")

    # Create a colorbar
    cbar = fig.colorbar(c, ax=ax, orientation='vertical', extend='both')
    cbar.set_label('Seasonal precipitation\n(standardised anomaly)')

    # Save the figure as a PNG file
    plt.savefig(os.path.join(output_dir, f"{year}.png"))
    plt.close(fig)

