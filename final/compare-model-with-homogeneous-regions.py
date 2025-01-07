from functions import *
import matplotlib.pyplot as plt
from cartopy.feature import ShapelyFeature
from cartopy import crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import numpy as np
import colormaps as cmaps
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

cmap = cmaps.BlueWhiteOrangeRed_r.discrete(17)

gdf = gpd.read_file("/home/users/rz908899/geodata/india_states_lores.zip")
region_polys = plot_homogeneous_regions(None, gdf, plot=False)

region_list = ['1-nmi', '2-nwi', '3-nci', '4-nei', '5-wpi', '6-epi', '7-spi']

region_prcp_dfs = {region:prepare_prcp_df(region) for region in region_list}

imd_ensemble_mean, lons, lats, year_list = fetch_cnn_ensemble('IMD')
era_ensemble_mean, lons, lats, year_list = fetch_cnn_ensemble('ERA5')

years_to_plot = [1895,1897,1896]

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(8.5, 10), subplot_kw={'projection': ccrs.PlateCarree()})

cook_data = xr.open_dataset("../cook-paleo-data.nc", decode_times=False)
cook_data = cook_data.transpose('T', 'Y', 'X')
cook_years = np.arange(1300,2006)
print(cook_data)

for col, year in enumerate(years_to_plot):
	# IMD Ensemble mean
	idt_imd = find(year, year_list)
	imd_data = imd_ensemble_mean[idt_imd]
	c = axes[1, col].pcolormesh(lons, lats, imd_data, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
	axes[0, col].set_title(f"{year}")
	
	mvlr_data = np.load(f"eof/{year}.npy")
	axes[2, col].pcolormesh(lons, lats, mvlr_data, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
	mask = mvlr_data<1e-10
	mask*= mvlr_data>-1e-10


	'''
	# ERA5 Ensemble mean
	idt_era = find(year, year_list)
	era_data = era_ensemble_mean[idt_era]*(1-mask)
	axes[2, col].pcolormesh(lons, lats, era_data, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
	'''
	
	idt_cook = find(year, cook_years)
	cook_prcp = cook_data['pdsi'].values[idt_cook]
	cook_prcp = RegularGridInterpolator((cook_data['Y'], cook_data['X']), cook_prcp, method='nearest')((lats[:,None], lons[None,:]))
	axes[3,col].pcolormesh(lons, lats, cook_prcp*(1-mask), cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)


	# Region polygons coloured by rainfall
	for (region, poly), code in zip(region_polys.items(), region_list):
		
		# Get corresponding color from dataframe
		prcp_val = region_prcp_dfs[code][region_prcp_dfs[code]['year'] == year]['summer_prcp'].values[0]
		
		prcp_normed = np.clip((prcp_val+1.5)/3, 0, 1)
		
		color = cmap(prcp_normed)  
		shape_feature = ShapelyFeature([poly], ccrs.PlateCarree(), facecolor=color, edgecolor='none')
		axes[0, col].add_feature(shape_feature)
		boundaries = ShapelyFeature([poly], ccrs.PlateCarree(), facecolor='none', edgecolor='k')
		
		for i in range(4):
	   		axes[i, col].add_feature(boundaries)


	#axes[2, col].set_title(f"{year}")

axes[0, 0].text(-0.2, 0.5, "(a) homogeneous region\nreconstruction\n(Sontakke et al., 2008)", rotation=90, va='center', ha='center', transform=axes[0, 0].transAxes)

axes[1, 0].text(-0.2, 0.5, "(b) CNN ensemble mean\n(IMD gridded gauges)", rotation=90, va='center', ha='center', transform=axes[1, 0].transAxes)

axes[3, 0].text(-0.2, 0.5, "(d) PCA regression PDSI\n(Cook et al., 2010)", rotation=90, va='center', ha='center', transform=axes[3, 0].transAxes)

axes[2, 0].text(-0.2, 0.5, "(c) PCA regression\n(IMD gridded gauge)", rotation=90, va='center', ha='center', transform=axes[2, 0].transAxes)

for ax in axes.ravel():
	ax.set_ylim([6.5, 38.5])
	ax.set_xlim([66.5, 100])

fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.5])
cbar = fig.colorbar(c, cax=cbar_ax, orientation='vertical', extend='both')
cbar.set_label('Standardised seasonal precipitation anomaly')
cbar.set_ticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])

plt.show()





