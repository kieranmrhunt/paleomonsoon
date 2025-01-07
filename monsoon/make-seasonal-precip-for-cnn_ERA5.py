import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# 1. Load both datasets
ds_imd = xr.open_dataset("../era5/IMD-gridded-1901-2020.nc")
ds_era5 = xr.open_dataset("../era5/era5-monthly-means-sf.nc").sel(expver=1)

# 2. Interpolate the ERA5 data onto the IMD grid
ds_era5_interpolated = ds_era5.interp(latitude=ds_imd.lat, longitude=ds_imd.lon)

# 3. Extract summer and winter data for ERA5
summer = [6,7,8,9]
summer_ds = ds_era5_interpolated.where(ds_era5_interpolated['time.month'].isin(summer), drop=True).resample(time="Y").mean("time")

# 4. Save the resampled summer data
summer_filtered = summer_ds.mtpr.values*3600*24
summer_filtered[summer_filtered<0] = 0
np.save("ERA5-JJAS-means.npy", summer_filtered)

