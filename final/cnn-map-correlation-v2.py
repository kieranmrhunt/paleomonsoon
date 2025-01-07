import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner as gridliner
import geopandas as gpd
import matplotlib.colors as mcolors
from functions import *
from scipy.ndimage import generic_filter

def corr2d(x, y, axis=0):
	x =	np.array(x)
	y = np.array(y)
	mx = np.mean(x, axis=axis, keepdims=True)
	my = np.mean(y, axis=axis, keepdims=True)
	sx = np.std(x, axis=axis, keepdims=True)
	sy = np.std(y, axis=axis, keepdims=True)
	
	r = np.mean((x-mx)*(y-my), axis=axis)/(sx*sy)
	return r.squeeze()

def median_filter(arr):
	smoothed_arr = generic_filter(arr, np.median, size=(1, 9, 9), mode='constant', cval=np.NaN, origin=0)
	return(smoothed_arr)

find = lambda x, arr: np.argmin(np.abs(x-arr))
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

source = 'IMD'
prcp_years, prcp_data = load_gridded_prcp(source=source)
era_years, era_data = load_gridded_prcp(source='ERA5')

df = pd.read_csv(f"ensemble-cnn/{source}-results.csv")

filtered_df = df[(df['pcc'] > 0.3) & (df['vd'] < 0.2)]
good_models = filtered_df['model_code'].tolist()

print(good_models)

start_year = prcp_years[0]
end_year = 1994
length = end_year-start_year+1

test_year_range = np.arange(start_year,end_year+1)
full_year_range = np.arange(1500,1995+1)

		

observed = []
predicted = []

for model_code in good_models:
	print(model_code)
	test_year = int(model_code.split("_")[-1][:-2])
	offset = int(model_code.split(".")[-1])
	val_years = np.array([(test_year-start_year+i+offset)%length for i in range(0,length,20)])+start_year
	
	years = np.r_[test_year, val_years]
	output = np.load(f"ensemble-cnn/output/{model_code}.npy")
	
	
	for year in years:
		idx_model = find(year, full_year_range)
		predicted.append(median_filter(output[idx_model].squeeze()))
		
		idx_obs = find(year, prcp_years)
		observed.append(median_filter(prcp_data[idx_obs].squeeze()))

corr = corr2d(observed, predicted)
print(corr.shape)


years = np.arange(1950,2020)
observed = []
predicted = []

for year in years:
	idx_obs = find(year, prcp_years)
	observed.append(median_filter(prcp_data[idx_obs].squeeze()))
	
	idx_model = find(year, era_years)
	predicted.append(median_filter(era_data[idx_model].squeeze()))

corr_era = corr2d(observed, predicted)
print(corr.shape)



gdf = gpd.read_file("/home/users/rz908899/geodata/mapindia/India_Boundary.shp")



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot for corr
ax1 = axes[0]
ax1.add_geometries(gdf.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='k', zorder=10)
pcm1 = ax1.pcolormesh(lons, lats, corr, vmin=0, vmax=1, cmap=plt.cm.terrain_r, transform=ccrs.PlateCarree())
ax1.set_title('(a) CNN ensemble mean vs IMD')
gl1 = ax1.gridlines(draw_labels=True, linewidth=0, color='none', alpha=0,)
gl1.xlabels_bottom = True
gl1.ylabels_left = True
gl1.xlabels_top = False
gl1.ylabels_right = False
gl1.xformatter = gridliner.LONGITUDE_FORMATTER
gl1.yformatter = gridliner.LATITUDE_FORMATTER

# Plot for corr_era
ax2 = axes[1]
ax2.add_geometries(gdf.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='k', zorder=10)
pcm2 = ax2.pcolormesh(lons, lats, corr_era, vmin=0, vmax=1, cmap=plt.cm.terrain_r, transform=ccrs.PlateCarree())
ax2.set_title('(b) ERA5 vs IMD')
gl2 = ax2.gridlines(draw_labels=True, linewidth=0, color='none', alpha=0,)
gl2.xlabels_bottom = True
gl2.ylabels_right = True
gl2.xlabels_top = False
gl2.ylabels_left = False
gl2.xformatter = gridliner.LONGITUDE_FORMATTER
gl2.yformatter = gridliner.LATITUDE_FORMATTER

# Colorbar
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.02])
cbar = fig.colorbar(pcm1, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label('Correlation coefficient')

plt.show()



