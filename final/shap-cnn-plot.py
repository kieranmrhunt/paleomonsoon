import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from functions import *
from glob import glob
import geopandas as gpd

fnames = glob("shap/cnn-shap-values/IMD_*.npy")

# Load the shap values
shap_values = [np.load(fname) for fname in fnames]

# Reshape the data: (17415, 5, 58) -> (129, 135, 5, 58)
reshaped_shap = np.reshape(shap_values, (-1, 129, 135, 5, 58))

# Take the absolute values
abs_shap = np.abs(reshaped_shap)
avg_shap = np.mean(abs_shap, axis=(0,3))

# Latitude and Longitude arrays
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

# Custom colour map
colours = ['white', 'lightgray', 'grey', 'yellow', 'orange', 'red']
custom_cmap = LinearSegmentedColormap.from_list('c_cmap', colours)

# Indices to plot in the last dimension
#indices_to_plot = [44,21,18]

indices_to_plot = [23, 12, 25, 54]
#23, 12, 25, 54

labels_to_plot  = ["(a) Tree ring, Gandaki, Nepal", 
				   "(b) Tree ring, Kerala, India", 
				   "(c) Tree ring, Ladakh", 
				   "(d) Speleothem, Hoq Cave, Yemen"]


max_idx = np.argsort(avg_shap.mean(axis=(0,1)))[::-1]
print(max_idx)

x1, x2 = 40, 95  
y1, y2 = -10, 40
min_resolution = 10
paleo_df, metadata = prepare_paleo_df(x1,x2,y1,y2,min_resolution, return_metadata=True)
#print(paleo_df.columns)
#print(metadata[['paleoData_TSid', 'geo_meanLon', 'geo_meanLat']])

gdf = gpd.read_file("/home/users/rz908899/geodata/ne_10m_admin_0_countries_ind.shp")

fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), subplot_kw={'projection': ccrs.PlateCarree()})
for i, idx in enumerate(indices_to_plot):
	ax = axes.ravel()[i]
	proxy_data = metadata.iloc[idx]
	print(proxy_data.paleoData_TSid)
	plot_data = avg_shap[:, :, idx]
	c = ax.pcolormesh(lons, lats, plot_data, cmap=custom_cmap, vmin=0, vmax=.1)
	
	ax.plot(proxy_data.geo_meanLon, proxy_data.geo_meanLat, 'co', mew=0.5, mec='k')
	ax.add_geometries(gdf.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=0.25, zorder=10)  
	ax.set_title(labels_to_plot[i])

# Add a colour bar
cbar = fig.colorbar(c, ax=axes.ravel().tolist(), orientation='vertical', pad=0.1)
cbar.set_label('Mean estimated Shapley value (standardised)')

plt.show()




