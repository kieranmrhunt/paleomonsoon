import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas
import colormaps as cmaps

cmap = cmaps.BlueWhiteOrangeRed_r.discrete(17)


source = 'IMD'


find = lambda x, arr: np.argmin(np.abs(x-arr))
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

prcp_data = np.load(f"../monsoon/{source}-JJAS-means.npy")
prcp_mean = prcp_data.mean(axis=0)
prcp_std = prcp_data.std(axis=0)

df = pd.read_csv(f"ensemble-cnn/{source}-results.csv")

filtered_df = df[(df['pcc'] > 0.3) & (df['vd'] < 0.2)]
good_models = filtered_df['model_code'].tolist()

#print(good_models)
print(len(good_models))

year_list = np.arange(1500,1996)


ensemble = []

for model_code in good_models:
	
	output = np.load(f"ensemble-cnn/output/{model_code}.npy").squeeze()	
	ensemble.append(output)
	print(model_code, end="\r")
	

ensemble_mean = np.mean(ensemble, axis=0)

famines = {1631:'Entire India',
		  #1668:'Bombay',
           1712:'Southern India\nCalcutta\nEast Coast',
           1759:'Kutch to Sindh\nPunjab\nTamil Nadu',
           1792:'Doji Bara'
           }


# Create figure and axes
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(9, 9), subplot_kw={'projection': ccrs.PlateCarree()})

# Read geospatial data for India
df = geopandas.read_file("/home/users/rz908899/geodata/mapindia/India_Boundary.shp")

# Loop through each famine
for i, famine_year in enumerate(famines):
	years_to_plot = np.arange(famine_year-2, famine_year+1)
	for j, year in enumerate(years_to_plot):
		ax = axes[j, i]
		idt = find(year, year_list)
		ensemble_mean_year = ensemble_mean[idt]

		idt = find(year, year_list)
		ensemble_mean_year = ensemble_mean[idt]
		
		actual_rainfall = (ensemble_mean_year*prcp_std)+prcp_mean
		deviation = (actual_rainfall/(prcp_mean+1e-5)-1)*100
		
		deviation[prcp_mean == 0] = 0
		ensemble_mean_year[prcp_mean == 0] = 0
		
		c = ax.pcolormesh(lons, lats, ensemble_mean_year, cmap=cmap, vmin=-1.5, vmax=1.5, transform=ccrs.PlateCarree(), rasterized=True)

		df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.25)
		ax.add_feature(cfeature.COASTLINE)

		ax.text(0.95, 0.95, f"{year}", va='top', ha='right', transform=ax.transAxes)
	
	axes[0, i].set_title(famines[famine_year])
	[spine.set_linewidth(1.5) for spine in axes[1, i].spines.values()]


axes[0, 0].text(-0.125, 0.5, "Preceding year", rotation=90, va='center', ha='center', transform=axes[0, 0].transAxes)

axes[1, 0].text(-0.125, 0.5, "First famine year", rotation=90, va='center', ha='center', transform=axes[1, 0].transAxes)

axes[2, 0].text(-0.125, 0.5, "Following year", rotation=90, va='center', ha='center', transform=axes[2, 0].transAxes)


# Add a colorbar
fig.subplots_adjust(bottom=0.2)
cbar_ax = fig.add_axes([0.25, 0.1, 0.5, 0.02])
cbar = fig.colorbar(c, cax=cbar_ax, orientation='horizontal', extend='both')
cbar.set_label('Standardised seasonal precipitation anomaly')
cbar.set_ticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])



top = axes[0, 0].get_position().y1
bottom = axes[2, 0].get_position().y0

start = (0.05, top)  # Arrow starting position (right, top)
end = (0.05, bottom)  # Arrow ending position (right, bottom)

# Draw the arrow
plt.annotate(
    '',  # No text: we'll add text separately
    xy=end, xycoords='figure fraction',
    xytext=start, textcoords='figure fraction',
    arrowprops=dict(
        arrowstyle="-|>",
        connectionstyle="arc3",  # Straight line
        color="black",  # Arrow color
        lw=2  # Line width
    )
)

# Add the text near the arrow, adjust the position and rotation as needed
plt.text(
    start[0] - 0.015,  # X position, adjust as necessary
    (start[1] + end[1]) / 2,  # Midpoint of the arrow's start and end in Y
    "Progression of drought conditions",  # Text to display
    rotation=90,  # Rotate the text
    va='center',  # Vertically center
    ha='center',  # Horizontally center
    transform=fig.transFigure,
)



plt.show()
