import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as mcolors
from functions import *


def corr2d(x, y, axis=0):
	x =	np.array(x)
	y = np.array(y)
	mx = np.mean(x, axis=axis, keepdims=True)
	my = np.mean(y, axis=axis, keepdims=True)
	sx = np.std(x, axis=axis, keepdims=True)
	sy = np.std(y, axis=axis, keepdims=True)
	
	r = np.mean((x-mx)*(y-my), axis=axis)/(sx*sy)
	return r.squeeze()
	


find = lambda x, arr: np.argmin(np.abs(x-arr))
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

source = 'IMD'
prcp_years, prcp_data = load_gridded_prcp(source=source)

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
		predicted.append(output[idx_model].squeeze())
		
		idx_obs = find(year, prcp_years)
		observed.append(prcp_data[idx_obs].squeeze())

corr = corr2d(observed, predicted)
print(corr.shape)

gdf = gpd.read_file("/home/users/rz908899/geodata/mapindia/India_Boundary.shp")

ax = plt.axes(projection=ccrs.PlateCarree())
ax.add_geometries(gdf.geometry, ccrs.PlateCarree(), facecolor='none', edgecolor='k', zorder=10)  

plt.pcolormesh(lons, lats, corr, vmin=0, vmax=1, cmap=plt.cm.terrain_r, transform=ccrs.PlateCarree())

cb = plt.colorbar()
cb.set_label("Correlation coefficient")
plt.show()
	
	
	



