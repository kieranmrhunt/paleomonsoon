import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas
import matplotlib.colors as mcolors

def make_colormap(colors):
	"""Create a colormap from a list of colors."""
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
prcp_data = (prcp_data-prcp_mean)/prcp_std

df = pd.read_csv(f"ensemble-cnn/{source}-results.csv")
filtered_df = df[(df['pcc'] > 0.2) & (df['vd'] < 0.2)]
sorted_df = filtered_df.sort_values(by='pcc', ascending=False)

print(sorted_df.head(20))

model_code = f'{source}_1905.1'





output = np.load(f"ensemble-cnn/output/{model_code}.npy").squeeze()	

#year_to_plot = int(model_code.split("_")[1].split(".")[0])
year_to_plot = 1946

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), subplot_kw={'projection': ccrs.PlateCarree()})

df = geopandas.read_file("/home/users/rz908899/geodata/india_states_lores.zip")


ax1, ax2 = axes

year_list = np.arange(1901,2020)
idt = find(year_to_plot, year_list)
obs_rainfall = prcp_data[idt]
cs1 = ax1.pcolormesh(lons, lats, obs_rainfall, cmap=newcmp, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree())


year_list = np.arange(1500,1996)
idt = find(year_to_plot, year_list)
model_rainfall = output[idt]
cs2 = ax2.pcolormesh(lons, lats, model_rainfall, cmap=newcmp, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree())


for ax in axes:
	

	df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.25)
	ax.add_feature(cfeature.COASTLINE)
	gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0, color='gray', alpha=0.5, linestyle='--')
	gl.xlabels_top = False
	gl.ylabels_right = False
	gl.xlines = False 
	gl.ylines = False
	gl.xformatter = LONGITUDE_FORMATTER
	gl.yformatter = LATITUDE_FORMATTER

ax1.set_title("(a) observed values (IMD gauges)")
ax2.set_title("(b) predicted values")


# Add a colorbar
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
cbar = fig.colorbar(cs1, cax=cbar_ax, orientation='vertical', extend='both')
cbar.set_label('Standardised precipitation')
cbar.set_ticks([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5])
plt.show()

