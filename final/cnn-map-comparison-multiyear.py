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
from functions import *
from scipy.ndimage import generic_filter
from scipy.stats import linregress
import colormaps as cmaps

#cmap = cmaps.hotcold_18lev
cmap = cmaps.BlueWhiteOrangeRed_r.discrete(17)

source = 'IMD'
start_year = 1901
length = 94
model_code = f'{source}_1905.1'

'''
source = 'ERA5'
model_code = f'{source}_1959.5'
'''

find = lambda x, arr: np.argmin(np.abs(x-arr))
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

prcp_years, prcp_data = load_gridded_prcp(source)

start_year = prcp_years[0]
length = 1995-start_year

df = pd.read_csv(f"ensemble-cnn/{source}-results.csv")
filtered_df = df[(df['pcc'] > 0.2) & (df['vd'] < 0.2)]
sorted_df = filtered_df.sort_values(by='pcc', ascending=False)

print(sorted_df.head(20))


test_year = int(model_code.split("_")[-1].split(".")[0])
j = int(model_code.split("_")[-1].split(".")[1])

years_to_plot = np.array([(test_year-start_year+i+j)%length for i in range(0,length,20)])+start_year
years_to_plot = sorted(years_to_plot)

print(years_to_plot)

output = np.load(f"ensemble-cnn/output/{model_code}.npy").squeeze()	



fig, axes = plt.subplots(nrows=4, ncols=len(years_to_plot), figsize=(10.5, 10.5), subplot_kw={'projection': ccrs.PlateCarree()})

df = geopandas.read_file("/home/users/rz908899/geodata/mapindia/India_Boundary.shp")

for ax, year_to_plot in zip(axes.T, years_to_plot):
	ax1, ax2, ax3, ax4 = ax

	year_list = prcp_years
	idt = find(year_to_plot, year_list)
	obs_rainfall = prcp_data[idt].squeeze()
	smoothed_obs = generic_filter(obs_rainfall, np.median, size=(9, 9), mode='constant', cval=np.NaN, origin=0)
	cs1 = ax1.pcolormesh(lons, lats, obs_rainfall, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
	mask = ~np.isnan(smoothed_obs)
	

	year_list = np.arange(1500,1996)
	idt = find(year_to_plot, year_list)
	model_rainfall = output[idt]
	smoothed_model = generic_filter(model_rainfall, np.median, size=(9, 9), mode='constant', cval=np.NaN, origin=0)
	cs2 = ax2.pcolormesh(lons, lats, model_rainfall, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
	_,_,r,_,_ = linregress(smoothed_obs[mask].ravel(), smoothed_model[mask].ravel())
	ax2.text(0.95, 0.95, "r={:1.2f}".format(r), ha='right', va='top', transform=ax2.transAxes)	
	
	
	mvlr_rainfall = np.load(f"mvlr/{year_to_plot}.npy")
	smoothed_mvlr = generic_filter(mvlr_rainfall, np.median, size=(9, 9), mode='constant', cval=np.NaN, origin=0)
	cs3 = ax3.pcolormesh(lons, lats, mvlr_rainfall, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
	_,_,r,_,_ = linregress(smoothed_obs[mask].ravel(), smoothed_mvlr[mask].ravel())
	ax3.text(0.95, 0.95, "r={:1.2f}".format(r), ha='right', va='top', transform=ax3.transAxes)
	
	eof_rainfall = np.load(f"eof/{year_to_plot}.npy")
	smoothed_eof = generic_filter(eof_rainfall, np.median, size=(9, 9), mode='constant', cval=np.NaN, origin=0)
	cs4 = ax4.pcolormesh(lons, lats, eof_rainfall, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
	_,_,r,_,_ = linregress(smoothed_obs[mask].ravel(), smoothed_eof[mask].ravel())
	ax4.text(0.95, 0.95, "r={:1.2f}".format(r), ha='right', va='top', transform=ax4.transAxes)
	
	
	ax1.set_title(year_to_plot)

	
	for axs in ax:
		df.plot(ax=axs, facecolor='none', edgecolor='k', linewidth=0.5)
		axs.add_feature(cfeature.COASTLINE)
		axs.set_xlim([67,98])
		axs.set_ylim([6,39])


axes[0,0].text(-0.2, 0.5, f'(a) observed values\n({source})', va='bottom', ha='center',
               rotation='vertical', rotation_mode='anchor',
               transform=axes[0,0].transAxes)

axes[1,0].text(-0.2, 0.5, '(b) predicted values\n(CNN model)', va='bottom', ha='center',
               rotation='vertical', rotation_mode='anchor',
               transform=axes[1,0].transAxes)

axes[2,0].text(-0.2, 0.5, '(c) predicted values\n(linear regression)', va='bottom', ha='center',
               rotation='vertical', rotation_mode='anchor',
               transform=axes[2,0].transAxes)

axes[3,0].text(-0.2, 0.5, '(d) predicted values\n(PCA regression)', va='bottom', ha='center',
               rotation='vertical', rotation_mode='anchor',
               transform=axes[3,0].transAxes)


# Add a colorbar
fig.subplots_adjust(left=.1,bottom=0.2, wspace=0)
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
cbar = fig.colorbar(cs1, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label('Standardised seasonal precipitation anomaly')
cbar.set_ticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
plt.show()

