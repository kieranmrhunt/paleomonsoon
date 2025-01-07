import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from tensorflow.keras.models import load_model
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import geopandas as gpd

from matplotlib.colors import LinearSegmentedColormap
colours = ['white', 'lightgray', 'grey', 'yellow', 'orange', 'red']
custom_cmap = LinearSegmentedColormap.from_list('c_cmap', colours)


datasets = {'1-nmi':'NMI', '2-nwi':'NWI', '3-nci':'NCI', 
            '4-nei':'NEI', '5-wpi':'WPI', '6-epi':'EPI', 
            '7-spi':'SPI', 'aismr':'AI'}

# Prepare paleo data
x1, x2 = 40, 120  
y1, y2 = -10, 50
min_resolution = 10
paleo_df, metadata = prepare_paleo_df(x1, x2, y1, y2, min_resolution, return_metadata=True)

# Extract full dataset for SHAP value computation
X_full = paleo_df[(paleo_df['year'] >= 1500) & (paleo_df['year'] < 1995)].drop(columns='year').values[:, 1:]

fig, axes = plt.subplots(nrows=3, ncols=3, subplot_kw={'projection': ccrs.PlateCarree()},
                         figsize=(10,8))

gdf = gpd.read_file("/home/users/rz908899/geodata/india_states_lores.zip")
region_polys = plot_homogeneous_regions(None, gdf, plot=False)
print(region_polys)

for i, dataset in enumerate(datasets):
	ax = axes.flatten()[i]
	ax.add_feature(cfeature.COASTLINE, zorder=3)
	ax.add_feature(cfeature.BORDERS, linestyle=':', zorder=2)
	ax.set_title(datasets[dataset], zorder=1)

	if dataset != 'aismr':
		shape_feature = ShapelyFeature([region_polys[datasets[dataset]]], ccrs.PlateCarree(), facecolor='powderblue', edgecolor='none')
	
	else:
		shape_feature = ShapelyFeature(region_polys.values(), ccrs.PlateCarree(), facecolor='powderblue', edgecolor='none')
	
	ax.add_feature(shape_feature)

	# Read the metadata file and sort it
	df = pd.read_csv(f"shap/timelines/metadata-{dataset}.csv")
	df['score'] = df['linear_correlation_coefficient'] - df['KGE']
	df_sorted = df.sort_values('score', ascending=False)
	print(df_sorted)

	# Initialize an array to store aggregated SHAP values
	aggregated_shap_values = np.zeros(X_full.shape[1])

	# Loop over the top 5 models
	for j in range(5):
		model_data = df_sorted.iloc[j]
		model_name = model_data.model_label

		model_path = f'shap/timelines/models/{model_name}.h5'
		model = load_model(model_path, custom_objects={'loss': combined_loss(alpha=1., beta=1, gamma=2.)})

		# Compute SHAP values for the model
		e = shap.DeepExplainer(model, X_full)
		shap_values = np.array(e.shap_values(X_full)).squeeze()

		# Average and aggregate the SHAP values
		importance = np.mean(np.abs(shap_values), axis=0)
		aggregated_shap_values += importance

	# Take the average of aggregated SHAP values
	aggregated_shap_values /= 5
	idx = np.argsort(aggregated_shap_values)[::-1]


	# Extract geo-information
	lons = metadata['geo_meanLon'].values[1:]
	lats = metadata['geo_meanLat'].values[1:]

	# Plot the averaged feature importance
	sorted_indices = np.argsort(aggregated_shap_values)[::-1]
	lons_sorted = lons[sorted_indices]
	lats_sorted = lats[sorted_indices]
	aggregated_shap_values_sorted = aggregated_shap_values[sorted_indices]

	# Plot the points
	for lon, lat, importance in zip(lons_sorted, lats_sorted, aggregated_shap_values_sorted):
		sc = ax.scatter(lon, lat, c=[importance], s=40, cmap=custom_cmap, edgecolors='black', linewidths=0.5, zorder=importance+10, vmin=0, vmax=0.15)

axes.flatten()[-1].remove()

fig.subplots_adjust(bottom=0.15, wspace = .15, hspace = .18)

cbar_ax = fig.add_axes([0.2, 0.08, 0.6, 0.02])
cb = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal', extend='max')
cb.set_label("Mean Shapley value magnitude (standardised)")

plt.show()
