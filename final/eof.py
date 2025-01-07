from functions import *
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import geopandas
import cartopy.feature as cfeature
from eofs.standard import Eof

# Load your gridded precipitation data
source = 'IMD'
prcp_years, prcp_data = load_gridded_prcp(source=source)

find = lambda x, arr: np.argmin(np.abs(x-arr))
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

# Define region and resolution for your paleoclimate data frame
x1, x2 = 40, 120  
y1, y2 = -10, 50
min_resolution = 10
paleo_df = prepare_paleo_df(x1, x2, y1, y2, min_resolution)

#test_year_lists = [np.arange(0,100,20)+1901+i for i in range(0,20)]
test_year_lists = [(1906,1926,1946,1966,1986),]

for test_years in test_year_lists:
	print(test_years)

	# Specify the years to withhold for testing
	start_year = prcp_years[0]
	#test_years = [1906, 1926, 1946, 1966, 1986]  # Example years, replace with your choice
	val_years = []
	train_years = [y for y in range(start_year, 1995) if (y not in test_years) and (y not in val_years)]
	extended_years = np.arange(1500,1902)

	X_train, X_val, X_test, X_extended, y_train, y_test, y_val = split_train_val_test(paleo_df, prcp_data, prcp_years, train_years, val_years, test_years)
	print(len(X_extended))

	solver = Eof(y_train.squeeze())
	pcs = np.array(solver.pcs(npcs=8))
	eofs = solver.eofs(neofs=8)

	print(pcs.shape)
	print(X_train.shape)


	pc_lrs = [LinearRegression().fit(X_train.squeeze(), pc) for pc in pcs.T]

	# Preparing the output prediction array
	pred_shape = [len(test_years), y_test.shape[-3], y_test.shape[-2]]  # This should be something like (num_years, 129, 135)
	y_pred = np.zeros(pred_shape)  # Initialize the array that will hold your predictions


	for iX, X in enumerate(X_test):
		for iP, pc_lr in enumerate(pc_lrs):
			pc = pc_lr.predict(X[None,:])
			y_pred[iX] += eofs[iP]*pc

	y_pred = y_pred[:,:,:,None]

	for rain, year in zip(y_pred, test_years):
		np.save(f"eof/{year}.npy", rain.squeeze())



years_to_plot = np.array(test_years)

fig, axes = plt.subplots(nrows=2, ncols=len(years_to_plot), figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
df = geopandas.read_file("/home/users/rz908899/geodata/india_states_lores.zip")


# Convert year numbers to indices in your arrays
year_indices = [np.where(prcp_years == year)[0][0] for year in years_to_plot]

for i, (ax, year, idx) in enumerate(zip(axes.T, years_to_plot, year_indices)):
    ax1, ax2 = ax

    # Observed rainfall for the year
    obs_rainfall = prcp_data[idx, :, :, 0]  # Update indexing based on your data's shape
    cs1 = ax1.pcolormesh(lons, lats, obs_rainfall, cmap='RdBu', vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree())

    # Predicted rainfall for the year
    # Note: Adjust indexing based on how y_pred is structured in your script
    model_rainfall = y_pred[i, :, :, 0]  # Update this based on your predicted data's shape
    cs2 = ax2.pcolormesh(lons, lats, model_rainfall, cmap='RdBu', vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree())

    

    ax1.set_title(f'Year {year} Observed')
    ax2.set_title(f'Year {year} Predicted')
    
    for axs in ax:
        df.plot(ax=axs, facecolor='none', edgecolor='k', linewidth=0.25)
        axs.add_feature(cfeature.COASTLINE)
        axs.set_xlim([67, 98])
        axs.set_ylim([6, 39])

# Setting common labels
axes[0,0].text(-0.2, 0.5, 'Observed values\n(IMD)', va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=axes[0,0].transAxes)
axes[1,0].text(-0.2, 0.5, 'Predicted values', va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=axes[1,0].transAxes)

# Add a colorbar
fig.subplots_adjust(left=.1, bottom=0.25, wspace=0.4, hspace=0.4)
cbar_ax = fig.add_axes([0.25, 0.15, 0.5, 0.03])  # Adjust the position as necessary
cbar = fig.colorbar(cs1, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label('Standardised seasonal precipitation anomaly')
cbar.set_ticks([-1.5, -0.9, -0.3, 0.3, 0.9, 1.5])

plt.show()

