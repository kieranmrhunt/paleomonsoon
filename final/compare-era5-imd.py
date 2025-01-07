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
from scipy.stats import linregress

find = lambda x, arr: np.argmin(np.abs(x-arr))
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

prcp_years, prcp_data = load_gridded_prcp(source='IMD')
era_years, era_data = load_gridded_prcp(source='ERA5')

year_to_plot = 1991

imd_prcp = prcp_data[find(year_to_plot, prcp_years)].squeeze()
era_prcp = era_data[find(year_to_plot, era_years)].squeeze()

smoothed_obs = generic_filter(imd_prcp, np.median, size=(9, 9), mode='constant', cval=np.NaN, origin=0)
mask = ~np.isnan(smoothed_obs)
smoothed_model = generic_filter(era_prcp, np.median, size=(9, 9), mode='constant', cval=np.NaN, origin=0)
_,_,r,_,_ = linregress(smoothed_obs[mask].ravel(), smoothed_model[mask].ravel())

print(r)



gdf = gpd.read_file("/home/users/rz908899/geodata/mapindia/India_Boundary.shp")



fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

# Plot for corr
ax1 = axes[0]
ax1.add_geometries(gdf.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='k', zorder=10)
pcm1 = ax1.pcolormesh(lons, lats, imd_prcp, vmin=-1.5, vmax=1.5, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
ax1.set_title('(a) IMD')
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
pcm2 = ax2.pcolormesh(lons, lats, era_prcp, vmin=-1.5, vmax=1.5, cmap=plt.cm.RdBu_r, transform=ccrs.PlateCarree())
ax2.set_title('(b) ERA5')
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



