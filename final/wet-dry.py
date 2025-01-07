from functions import *
import matplotlib.pyplot as plt
from cartopy.feature import ShapelyFeature
from cartopy import crs as ccrs
from matplotlib.colors import LinearSegmentedColormap
import geopandas as gpd
import numpy as np
import colormaps as cmaps
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

df = gpd.read_file("/home/users/rz908899/geodata/mapindia/India_Boundary.shp")

cmap = cmaps.BlueWhiteOrangeRed_r.discrete(17)

find = lambda x, arr: np.argmin(np.abs(x-arr))
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)


cnn_ensemble_prcp, lons, lats, cnn_year_list = fetch_cnn_ensemble('IMD', return_all_models = True)
cnn_ensemble_mean_prcp = np.mean(cnn_ensemble_prcp, axis=0)
year_mask = cnn_year_list<=1900
cnn_ensemble_mean_prcp = cnn_ensemble_mean_prcp[year_mask]
cnn_year_list = cnn_year_list[year_mask]

imd_year_list, imd_prcp = load_gridded_prcp('IMD')
mask = ~np.isnan(imd_prcp.mean(axis=(0,3)))[None,:,:]
imd_prcp = imd_prcp.squeeze()

print(cnn_ensemble_prcp.shape)
cnn_prcp_timeseries = standardise(np.nanmean(destandardise(cnn_ensemble_mean_prcp),axis=(1,2)))
cnn_prcp_all_models = [standardise(np.nanmean(destandardise(model_prcp),axis=(1,2))) for model_prcp in cnn_ensemble_prcp]
cnn_upper_bound = np.nanmax(cnn_prcp_all_models, axis=0)[year_mask]
cnn_lower_bound = np.nanmin(cnn_prcp_all_models, axis=0)[year_mask]


imd_prcp_timeseries = standardise(np.nanmean(destandardise(imd_prcp),axis=(1,2)))

cnn_rank = np.argsort(cnn_prcp_timeseries)
imd_rank = np.argsort(imd_prcp_timeseries)



fig = plt.figure(figsize=(10, 5))
gs = GridSpec(7, 9, figure=fig)

cnn_axes = [fig.add_subplot(gs[0:2,i], projection=ccrs.PlateCarree()) for i in [0,1,2,3,5,6,7,8]]
idxs = [0,1,2,3,-4,-3,-2,-1]


for idx, ax in zip(idxs, cnn_axes):
	
	idt = cnn_rank[idx]
	prcp = cnn_ensemble_mean_prcp[idt]
	year = cnn_year_list[idt]

	ax.pcolormesh(lons, lats, prcp, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
	ax.set_title(year, fontsize='small')
	df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.5)
	ax.set_ylim([6.5, 38.5])
	ax.set_xlim([66.5, 100])

imd_axes = [fig.add_subplot(gs[2:4,i], projection=ccrs.PlateCarree()) for i in [0,1,2,6,7,8]]
idxs = [0,1,2,-3,-2,-1]

for idx, ax in zip(idxs, imd_axes):
	
	idt = imd_rank[idx]
	prcp = imd_prcp[idt]
	year = imd_year_list[idt]

	c = ax.pcolormesh(lons, lats, prcp, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)
	ax.set_title(year, fontsize='small')
	df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.5)
	ax.set_ylim([6.5, 38.5])
	ax.set_xlim([66.5, 100])



ax = fig.add_subplot(gs[-2:,:])
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

all_years = np.concatenate([cnn_year_list, imd_year_list])
all_prcp = np.concatenate([cnn_prcp_timeseries, imd_prcp_timeseries])

data = {'Year': all_years, 'Precipitation': all_prcp}
df_csv = pd.DataFrame(data)

output_path = "all_years_precipitation.csv"
df_csv.to_csv(output_path, index=False)

ax.bar(all_years, all_prcp, color=cmaps.BlueYellowRed_r(rescale(cnn_prcp_timeseries)))
ax.fill_between(cnn_year_list, cnn_lower_bound, cnn_upper_bound, color='lightgrey', zorder=0)


lowpass_rainfall = gaussian_filter(all_prcp, 10)
ax.plot(all_years, lowpass_rainfall*3, color='k', ls=':', lw=0.5)
ax.axvline(1900, color = 'g', ls= '--')

ax.set_xlim([1500,2023])
ax.set_title("(c) Ensemble mean all-India seasonal rainfall (standardised anomaly)")



cnn_axes[0].text(-0.2, 0.5, "(a) model", rotation=90, va='center', ha='center', transform=cnn_axes[0].transAxes)

imd_axes[0].text(-0.2, 0.5, "(b) IMD gauges", rotation=90, va='center', ha='center', transform=imd_axes[0].transAxes)

fig.subplots_adjust(right=0.85)

x0, x1 = cnn_axes[3].get_position().x1, cnn_axes[4].get_position().x0
y0, y1 = imd_axes[0].get_position().y0, imd_axes[0].get_position().y1
y2 = cnn_axes[0].get_position().y1

offset = 0.02
plt.text((x0+x1)/2+offset, (y0+y1)/2, "wettest\u27F6\nmonsoons", transform=fig.transFigure, ha='left')
plt.text((x0+x1)/2-offset, (y0+y1)/2, "\u27F5driest\nmonsoons", transform=fig.transFigure, ha='right')


cbar_ax = fig.add_axes([0.875, y0, 0.025, y2-y0])
cbar = fig.colorbar(c, cax=cbar_ax, orientation='vertical', extend='both')
cbar.set_label('Seasonal precipitation\n(standardised anomaly)')
cbar.set_ticks([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5])


plt.show()





