from functions import *
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import geopandas
import cartopy.feature as cfeature

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

# Specify the years to withhold for testing
start_year = prcp_years[0]
test_years = [1906, 1926, 1946, 1966, 1986]  # Example years, replace with your choice

test_year_lists = [np.arange(0,80,20)+1901+i for i in range(15,20)]

for test_years in test_year_lists:

	val_years = []
	train_years = [y for y in range(start_year, 1995) if (y not in test_years) and (y not in val_years)]

	X_train, X_val, X_test, X_extended, y_train, y_test, y_val = split_train_val_test(paleo_df, prcp_data, prcp_years, train_years, val_years, test_years)
	print(len(X_extended))

	# Preparing the output prediction array
	pred_shape = [len(test_years), y_test.shape[-3], y_test.shape[-2], y_test.shape[-1]]  # This should be something like (num_years, 129, 135)
	y_pred = np.zeros(pred_shape)  # Initialize the array that will hold your predictions

	# Loop through each grid point
	for i in range(pred_shape[1]):  # Loop over latitude
		print(i)
		for j in range(pred_shape[2]):  # Loop over longitude
		    # Get the training and test sets for the current grid point
		    y_train_gp = y_train[:, i, j, 0]  # Assuming y_train has a shape (num_years, lat, lon, 1)
		    y_test_gp = y_test[:, i, j, 0]  # Similarly for y_test
		    #print(y_train_gp)
		    #print(y_test_gp)

		    # Check if there is actual data to train on for this grid point
		    if not np.all(np.isnan(y_train_gp)):
		        # Train the model for the current grid point
		        model = LinearRegression()
		        model.fit(X_train, y_train_gp)

		        # Predict using the model for the current grid point
		        y_pred[:, i, j, 0] = model.predict(X_test)
		        #print(model.predict(X_test))

	#mvlr_rainfall_full = model.predict(X_train)

	for rain, year in zip(y_pred, test_years):
		np.save(f"mvlr/{year}.npy", rain.squeeze())


'''
years_to_plot = np.array(test_years)

fig, axes = plt.subplots(nrows=2, ncols=len(years_to_plot), figsize=(15, 10), subplot_kw={'projection': ccrs.PlateCarree()})
df = geopandas.read_file("/home/users/rz908899/geodata/india_states_lores.zip")

# You might need a custom colormap for your data
# newcmp = plt.get_cmap('RdBu')  # Example colormap, replace with your actual colormap

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
'''
