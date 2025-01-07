import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
import pandas as pd
import geopandas as gpd
from netCDF4 import Dataset
import xarray as xr
import numpy.ma as ma
from functions import *
import matplotlib.colors as mcolors

import warnings
from shapely.errors import ShapelyDeprecationWarning

# Suppress the ShapelyDeprecationWarning
warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)

x1, x2 = 40, 120  
y1, y2 = -10, 50  
min_resolution = 10

metadata_paths = [
    "../pages2k/metadata.csv",
    "../iso2k/metadata.csv",
    "../hrnh2k/temp_proxies/metadata.csv",
    "../hrnh2k/hydro_proxies/metadata.csv"
]

# Create an empty DataFrame to store the merged metadata
dfs = []

for path in metadata_paths:
    metadata_df = pd.read_csv(path)
    dfs.append(metadata_df)

merged_metadata = pd.concat(dfs, ignore_index=True)

# Load the datasets associated with the metadata paths
pages2k_df = pd.read_csv("../pages2k/values.csv")
iso2k_df = pd.read_csv("../iso2k/values.csv")
hrnh_temp_df = pd.read_csv("../hrnh2k/temp_proxies/values.csv")
hrnh_hydro_df = pd.read_csv("../hrnh2k/hydro_proxies/values.csv")

merged_data = pd.merge(pages2k_df, iso2k_df, on='year', how='outer')
merged_data = pd.merge(merged_data, hrnh_temp_df, on='year', how='outer')
merged_data = pd.merge(merged_data, hrnh_hydro_df, on='year', how='outer')


filtered_metadata = merged_metadata[
    (merged_metadata['geo_meanLon'] >= x1) & (merged_metadata['geo_meanLon'] <= x2) &
    (merged_metadata['geo_meanLat'] >= y1) & (merged_metadata['geo_meanLat'] <= y2) &
    (merged_metadata['resolution'] < min_resolution)
]


# Filter rows based on the year range
filtered_data = merged_data[(merged_data['year'] >= 1500) & (merged_data['year'] <= 1995)]

# Find valid ids (those that have at least one non-NaN value in the filtered data)
valid_ids = filtered_data.columns[filtered_data.notna().all()].tolist()

# Filter the metadata to only include these valid ids
filtered_metadata = filtered_metadata[filtered_metadata['paleoData_TSid'].isin(valid_ids)]

print(len(filtered_metadata))



# Extract the required data from filtered_metadata
lons = filtered_metadata['geo_meanLon'].values
lats = filtered_metadata['geo_meanLat'].values
proxy_types = filtered_metadata['archiveType'].values
resolutions = filtered_metadata['resolution'].values

print(np.unique(proxy_types))

# Define a mapping for proxy types
proxy_mapping = {
    "tree": "Tree", "Wood": "Tree", "TRE": "Tree",
    "marine sediment": "Marine Sediment", "SEA": "Marine Sediment",
    "coral": "Coral", "Coral": "Coral",
    "glacier ice": "Glacier Ice", "GlacierIce": "Glacier Ice",
    "Speleothem": "Speleothem", "SPE": "Speleothem",
    "LAK": "Lake",
    "OTH": "Other",
    "DOC": "Document"
}

# Update proxy_types with standardized names
proxy_types = [proxy_mapping.get(pt, 'Other') if pd.notna(pt) else 'Other' for pt in proxy_types]

# Define a color mapping based on the nature of the proxy types
color_mapping = {
    "Tree": "lightgreen",
    "Marine Sediment": "blue",
    "Coral": "coral",
    "Glacier Ice": "cyan",
    "Speleothem": "tomato",
    "Lake": "darkblue",
    "Other": "black",
    "Document": "grey"
}

# For sizing the points inversely by resolution
sizes = 1 / resolutions * 1000  # Adjust the scaling factor (1000) as needed

# Create a color map for proxy types
unique_proxy_types = list(set(proxy_types))
colors = plt.cm.jet(np.linspace(0, 1, len(unique_proxy_types)))
color_map = dict(zip(unique_proxy_types, colors))

# Create the map
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15,10))

gdf = gpd.read_file("/home/users/rz908899/geodata/india_states_lores.zip")
plot_homogeneous_regions(ax, gdf)


with Dataset('../data/era5-orography.nc') as ds:
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    orography = ds.variables['z'][0] #* ds.variables['z'].scale_factor + ds.variables['z'].add_offset

# Convert geopotential to altitude in meters (assuming standard gravitational acceleration)
altitude = orography / 9.80665 /1000

cmap = plt.cm.gray_r  # B&W colormap
cs = ax.pcolormesh(lon, lat, altitude, transform=ccrs.PlateCarree(), cmap=cmap, shading='auto', vmin=0, rasterized=True)


orig_blues = plt.cm.Blues
new_blues = mcolors.ListedColormap(orig_blues(np.linspace(0.25, 1, 256)))


ds = xr.open_dataset('../era5/era5-global-monthly-means.nc')
mtpr = ds['mtpr']
mean_mtpr = mtpr.mean(dim='time')*3600*24
contour_levels = np.arange(5,50,5)
cs2 = ax.contourf(ds['longitude'], ds['latitude'], mean_mtpr, contour_levels, transform=ccrs.PlateCarree(), cmap=new_blues, extend='max', rasterized=True)

# Plot each dataset location
for lon, lat, proxy_type, size in zip(lons, lats, proxy_types, sizes):
    ax.scatter(lon, lat, color=color_mapping[proxy_type], label=proxy_type, edgecolors='k', linewidth=0.5, zorder = 20)


ax.add_feature(cfeature.COASTLINE.with_scale('10m'), linewidth=0.5, edgecolor='gray')

# To avoid duplicate labels in the legend
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper left')

ax.set_xlim([np.min(lons)-1, np.max(lons)+1])
ax.set_ylim([np.min(lats)-1, np.max(lats)+1])

# Add gridlines and labels
gl = ax.gridlines(draw_labels=True, x_inline=False, y_inline=False)
gl.xlines = False
gl.ylines = False

plt.subplots_adjust(bottom=0.2)

offset = 0.05
# Get the bounds of the main subplot
bounds = ax.get_position()
x0 = bounds.x0+offset  # Left edge of main subplot
x1 = bounds.x1-offset
width = x1-x0  # Width of main subplot

# Set the positions for the colorbars based on the main subplot
width_orography = width * 0.4
width_mtpr = width * 0.4

left_orography = x0
left_mtpr = x0 + width * 0.6

# Create a new axis for the orography colorbar and add the colorbar
cax_orography = fig.add_axes([left_orography, 0.1, width_orography, 0.02])  # left, bottom, width, height
cb_orography = plt.colorbar(cs, cax=cax_orography, orientation='horizontal', extend='max')
cb_orography.set_label('Orography (km)')

# Create a new axis for the mtpr colorbar and add the colorbar
cax_mtpr = fig.add_axes([left_mtpr, 0.1, width_mtpr, 0.02])  # left, bottom, width, height
cb_mtpr = plt.colorbar(cs2, cax=cax_mtpr, orientation='horizontal')
cb_mtpr.set_label('Precipitation (mm/day)')

ax.set_title('Dataset Locations by Proxy Type')
plt.show()
