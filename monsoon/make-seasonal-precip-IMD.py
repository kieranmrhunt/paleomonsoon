import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

ds = xr.open_dataset("../era5/IMD-gridded-1901-2020.nc")
summer = [6,7,8,9]
winter = [1,2,3]

summer_core_zone = [73, 86, 18, 28]
winter_core_zone = [70, 80, 30, 37]

#summer_core_zone = [50, 120, 0, 45]
#winter_core_zone = [50, 120, 0, 45]


summer_ds = ds.where(ds['time.month'].isin(summer), drop=True).resample(time="Y").mean("time")
winter_ds = ds.where(ds['time.month'].isin(winter), drop=True).resample(time="Y").mean("time")
	
fig, axs = plt.subplots(1,2, subplot_kw={'projection': ccrs.PlateCarree()})

lons = summer_ds.lon.values
lats = summer_ds.lat.values

axs[0].pcolormesh(lons, lats, summer_ds.rain.values.mean(axis=0), vmin=0, cmap=plt.cm.terrain_r)
axs[1].pcolormesh(lons, lats, winter_ds.rain.values.mean(axis=0), vmin=0, cmap=plt.cm.terrain_r)

for ax in axs:

	ax.set_extent([67, 98, 5, 40])
	ax.add_feature(cfeature.BORDERS, linestyle=':')
	ax.add_feature(cfeature.COASTLINE)
	ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='none')
	ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

x1, x2, y1, y2 = summer_core_zone	
axs[0].plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], 'r-')
summer_filtered = summer_ds.sel(lat=slice(y1,y2), lon=slice(x1,x2)).rain.values
summer_filtered[summer_filtered<0]=np.nan
summer_prcp = np.nanmean(summer_filtered, axis=(-1,-2))
print(summer_prcp)

x1, x2, y1, y2 = winter_core_zone	
axs[1].plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1], 'r-')
winter_filtered = winter_ds.sel(lat=slice(y1,y2), lon=slice(x1,x2)).rain.values
winter_filtered[winter_filtered<0]=np.nan
winter_prcp = np.nanmean(winter_filtered, axis=(-1,-2))

output = pd.DataFrame.from_dict({"year":np.arange(1901,2021), "summer_prcp":summer_prcp, "winter_prcp": winter_prcp})
output.to_csv("all_india_prcp_IMD.csv", index=False)

plt.show()
