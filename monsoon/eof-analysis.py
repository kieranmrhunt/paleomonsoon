import xarray as xr
from eofs.standard import Eof
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy

lat_bounds = slice(45, 0)
lon_bounds = slice(60, 100)


ds = xr.open_mfdataset("era5/*")

ds_filtered = ds.sel(time=ds['time.year'] < 2023)
ds_filtered = ds_filtered.sel(latitude=lat_bounds, longitude=lon_bounds, expver=1)
ds_yearly_avg = ds_filtered.resample(time='1Y').mean('time')

ds_yearly_avg['u850'] = ds_yearly_avg.u.sel(level=850).drop('level')
ds_yearly_avg['v850'] = ds_yearly_avg.v.sel(level=850).drop('level')
#ds_yearly_avg['u200'] = ds_yearly_avg.u.sel(level=200).drop('level')
#ds_yearly_avg['v200'] = ds_yearly_avg.v.sel(level=200).drop('level')

ds_yearly_avg = ds_yearly_avg.drop_vars(['u', 'v'])


variables_to_standardize = ['t2m', 'mtpr', 'u850', 'v850', ]#'u200', 'v200']
coeffs = {}

for var in variables_to_standardize:
	mean_val = ds_yearly_avg[var].mean('time', keep_attrs=True)
	std_dev = ds_yearly_avg[var].std('time', keep_attrs=True)
	
	coeffs[var] = (mean_val, std_dev)

	ds_yearly_avg[var] = (ds_yearly_avg[var] - mean_val) / std_dev


t2m = ds_yearly_avg['t2m'].values
mtpr = ds_yearly_avg['mtpr'].values
u850 = ds_yearly_avg['u850'].values
v850 = ds_yearly_avg['v850'].values
#u200 = ds_yearly_avg['u200'].values
#v200 = ds_yearly_avg['v200'].values

combined = np.concatenate([t2m,mtpr,u850,v850], axis=-1)

print(combined.shape)


lons = ds_yearly_avg.longitude.values
lats = ds_yearly_avg.latitude.values
coslat = np.cos(np.deg2rad(lats))
wgts = np.sqrt(coslat)[:, np.newaxis]
solver = Eof(combined, weights=wgts)

eofs = solver.eofs(neofs=8, eofscaling=1)
pcs = solver.pcs(npcs=8, pcscaling=1)

print(solver.varianceFraction(neigs=8))

print(pcs.shape)
print(eofs.shape)

pc_df = pd.DataFrame.from_dict({"year":np.arange(1940,2023)})
for i in range(1,9):
	pc_df[f"pc{i}"] = pcs[:,i-1]
	
print(pc_df)

pc_df.to_csv("annual_pca_df.csv")

###plot first four eofs

fig, axes = plt.subplots(2,4, subplot_kw={"projection":cartopy.crs.PlateCarree()})
s = 10

for i, ax in enumerate(axes.ravel()):
	
	eofi = eofs[i]
		
	t2m, mtpr, u850, v850, = np.array_split(eofi, 4, axis=-1)
	
	mtpr = mtpr*coeffs['mtpr'][1]
	u850 = u850*coeffs['u850'][1]
	v850 = v850*coeffs['v850'][1]
	
	ax.pcolormesh(lons, lats, mtpr,)# vmin=-3, vmax=3)
	ax.quiver(lons[::s], lats[::s], u850[::s,::s], v850[::s,::s])

plt.show()
	





























