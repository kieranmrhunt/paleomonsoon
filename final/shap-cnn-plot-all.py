import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
from functions import *
from glob import glob
import os

# Create output folder if it doesn't exist
if not os.path.exists('cnn-shap-figs'):
	os.makedirs('cnn-shap-figs')

# Load the shap values
fnames = glob("shap/cnn-shap-values/IMD_*.npy")
shap_values = [np.load(fname) for fname in fnames]

# Reshape and average the data
reshaped_shap = np.reshape(shap_values, (-1, 129, 135, 5, 58))
abs_shap = np.abs(reshaped_shap)
avg_shap = np.mean(abs_shap, axis=(0,3))

# Latitude and Longitude arrays
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

# Custom colour map
colours = ['white', 'lightgray', 'grey', 'yellow', 'orange', 'red']
custom_cmap = LinearSegmentedColormap.from_list('c_cmap', colours)

# Prepare paleo data
x1, x2, y1, y2, min_resolution = 40, 95, -10, 40, 10
paleo_df, metadata = prepare_paleo_df(x1, x2, y1, y2, min_resolution, return_metadata=True)

# Iterate through indices in steps of 3
for start_idx in range(0, avg_shap.shape[2], 3):
	indices_to_plot = list(range(start_idx, start_idx + 3))
	print(indices_to_plot)

	fig, axes = plt.subplots(1, len(indices_to_plot), figsize=(15, 5), subplot_kw={'projection': ccrs.PlateCarree()})
	
	for i, idx in enumerate(indices_to_plot):
		if idx >= avg_shap.shape[2]:
			break
		
		ax = axes[i]
		proxy_data = metadata.iloc[idx]
		plot_data = avg_shap[:, :, idx]
		
		c = ax.pcolormesh(lons, lats, plot_data, cmap=custom_cmap, vmin=0, vmax=.1)
		ax.plot(proxy_data.geo_meanLon, proxy_data.geo_meanLat, 'kx')
		ax.coastlines()
		ax.set_title(f"Index {idx}")

	# Add a colour bar
	cbar = fig.colorbar(c, ax=axes.ravel().tolist(), orientation='horizontal', pad=0.1)
	cbar.set_label('SHAP Value')
	
	# Save the figure
	plt.savefig(f"cnn-shap-figs/shap_indices_{start_idx}_{start_idx+2}.png")

	plt.close(fig)

