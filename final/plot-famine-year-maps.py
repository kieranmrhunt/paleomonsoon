import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas
import matplotlib.colors as mcolors

def make_colormap(colors):
	return mcolors.LinearSegmentedColormap.from_list("", colors)
   
# Define the individual colormaps
red_cmap = make_colormap([(1, 0.8, 0.8), (0.8, 0, 0)])
yellow_cmap = make_colormap([(1, 1, 0.5), (1, 0.65, 0)])
white_cmap = make_colormap([(1, 1, 1), (1, 1, 1)])
green_cmap = make_colormap([(0.5, 1, 0.5), (0, 0.5, 0)])
blue_cmap = make_colormap([(0.5, 0.5, 1), (0, 0, 1)])

# Now, combine them
n = 32
newcolors = np.vstack((
	yellow_cmap(np.linspace(1, 0, n)),
	red_cmap(np.linspace(1, 0, n)),
	white_cmap(np.linspace(0, 1, n)),
	blue_cmap(np.linspace(0, 1, n)),
	green_cmap(np.linspace(0, 1, n))
))
newcmp = mcolors.ListedColormap(newcolors)


source = 'IMD'


find = lambda x, arr: np.argmin(np.abs(x-arr))
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

prcp_data = np.load(f"../monsoon/{source}-JJAS-means.npy")
prcp_mean = prcp_data.mean(axis=0)
prcp_std = prcp_data.std(axis=0)

df = pd.read_csv(f"ensemble-cnn/{source}-results.csv")

filtered_df = df[(df['pcc'] > 0.3) & (df['vd'] < 0.2)]
good_models = filtered_df['model_code'].tolist()

#print(good_models)
print(len(good_models))

year_list = np.arange(1500,1996)


ensemble = []

for model_code in good_models:
	
	output = np.load(f"ensemble-cnn/output/{model_code}.npy").squeeze()	
	ensemble.append(output)
	print(model_code, end="\r")
	

ensemble_mean = np.mean(ensemble, axis=0)

famines = {1631:'Entire India',
		  #1668:'Bombay',
           1712:'Southern India\nCalcutta\nEast Coast',
           1759:'Kutch to Sindh\nPunjab\nTamil Nadu',
           1792:'Doji Bara'
           }


# Create figure and axes
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8, 20), subplot_kw={'projection': ccrs.PlateCarree()})

# Read geospatial data for India
df = geopandas.read_file("/home/users/rz908899/geodata/india_states_lores.zip")

# Loop through each famine
for i, famine_year in enumerate(famines):
	years_to_plot = np.arange(famine_year-2, famine_year+1)
	for j, year in enumerate(years_to_plot):
		ax = axes[i, j]
		idt = find(year, year_list)
		ensemble_mean_year = ensemble_mean[idt]

		idt = find(year, year_list)
		ensemble_mean_year = ensemble_mean[idt]
		
		actual_rainfall = (ensemble_mean_year*prcp_std)+prcp_mean
		deviation = (actual_rainfall/(prcp_mean+1e-5)-1)*100
		
		deviation[prcp_mean == 0] = 0
		ensemble_mean_year[prcp_mean == 0] = 0
		
		c = ax.pcolormesh(lons, lats, ensemble_mean_year, cmap=newcmp, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree())

		df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.25)
		ax.add_feature(cfeature.COASTLINE)

		ax.text(0.95, 0.95, f"{year}", va='top', ha='right', transform=ax.transAxes)
	
	axes[i, 0].text(-0.2, 0.5, famines[famine_year], rotation=90, va='center', ha='center', transform=axes[i, 0].transAxes)


# Add a colorbar
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.02])
cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label('Standardised seasonal precipitation anomaly')
cbar.set_ticks([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5])

plt.show()
