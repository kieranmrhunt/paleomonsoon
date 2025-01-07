import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ds = xr.open_dataset("../era5/IMD-gridded-1901-2020.nc")
summer = [6,7,8,9]
winter = [1,2,3]


summer_ds = ds.where(ds['time.month'].isin(summer), drop=True).resample(time="Y").mean("time")
lons = summer_ds.lon.values
lats = summer_ds.lat.values

summer_filtered = summer_ds.rain.values
summer_filtered[summer_filtered<0]=0


print(summer_filtered.shape)

np.save("IMD-JJAS-means.npy", summer_filtered)
