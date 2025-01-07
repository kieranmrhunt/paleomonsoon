import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.stats import linregress

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



from sklearn.preprocessing import MinMaxScaler, StandardScaler


import random

find = lambda x, arr: np.argmin(np.abs(x-arr))

def set_seed(seed=1):
	import tensorflow as tf
	os.environ['PYTHONHASHSEED'] = str(seed)
	random.seed(seed)
	np.random.seed(seed)
	tf.random.set_seed(seed)
	


def load_gridded_prcp(source='IMD'):

	prcp_data = np.load(f"../monsoon/{source}-JJAS-means.npy")
	prcp_mean = np.nanmean(prcp_data,axis=0,keepdims=True)
	prcp_std = np.nanstd(prcp_data,axis=0,keepdims=True)

	prcp_data = (prcp_data-prcp_mean)/prcp_std
	prcp_data[np.isnan(prcp_data)] = 0

	prcp_data = prcp_data.reshape((prcp_data.shape[0], prcp_data.shape[1], prcp_data.shape[2], 1))
	if source=='IMD':
		prcp_years = np.arange(1901,2021)
	elif source=='ERA5':
		prcp_years = np.arange(1940,2024)
	
	return prcp_years, prcp_data


def destandardise(arr, source='IMD'):

	prcp_data = np.load(f"../monsoon/{source}-JJAS-means.npy")
	prcp_mean = np.nanmean(prcp_data,axis=0,keepdims=True)
	prcp_std = np.nanstd(prcp_data,axis=0,keepdims=True)
	
	nan_mask = ~np.isnan(prcp_data.mean(axis=0,keepdims=True))
	nan_mask[nan_mask==1]=np.nan

	arr_absolute = (arr*prcp_std)+prcp_mean
	arr_absolute*=nan_mask
	
	return arr_absolute



def standardise(arr):

	arr_mean = np.nanmean(arr,axis=0,keepdims=True)
	arr_std = np.nanstd(arr,axis=0,keepdims=True)

	arr_normed = (arr-arr_mean)/arr_std
	
	return arr_normed.squeeze()



def prepare_paleo_df(x1=40,x2=95,y1=-10,y2=40,min_res=10, add_shift=False, scale=True, return_metadata=False):
	
	pages2k_df = pd.read_csv("../pages2k/values.csv")
	iso2k_df = pd.read_csv("../iso2k/values.csv")
	hrnh_temp_df = pd.read_csv("../hrnh2k/temp_proxies/values.csv")
	hrnh_hydro_df = pd.read_csv("../hrnh2k/hydro_proxies/values.csv")

	paleo_df = pd.merge(pages2k_df, iso2k_df, on='year', how='outer')
	paleo_df = pd.merge(paleo_df, hrnh_temp_df, on='year', how='outer')
	paleo_df = pd.merge(paleo_df, hrnh_hydro_df, on='year', how='outer')

	metadata_paths = [
		"../pages2k/metadata.csv",
		"../iso2k/metadata.csv",
		"../hrnh2k/temp_proxies/metadata.csv",
		"../hrnh2k/hydro_proxies/metadata.csv"
	]

	# Create an empty DataFrame to store the merged metadata
	dfs = []

	for path in metadata_paths:
		metadata_df = pd.read_csv(path)
		dfs.append(metadata_df)

	merged_metadata = pd.concat(dfs, ignore_index=True)

	filtered_metadata = merged_metadata[
		(merged_metadata['geo_meanLon'] >= x1) & (merged_metadata['geo_meanLon'] <= x2) &
		(merged_metadata['geo_meanLat'] >= y1) & (merged_metadata['geo_meanLat'] <= y2) &
		#(merged_metadata['geo_meanLat'] + merged_metadata['geo_meanLon'] <=xy) &
		(merged_metadata['resolution'] < min_res)
	]


	filtered_ids = filtered_metadata['paleoData_TSid'].values
	filtered_columns = ['year'] + [col for col in filtered_ids if col in paleo_df.columns]
	paleo_df = paleo_df[filtered_columns]

	years_range = range(1500, 1995)
	paleo_df = paleo_df.loc[:, (paleo_df.loc[paleo_df['year'].isin(years_range)].notna().all())]

	#print(paleo_df)
	
	if add_shift:
		for col in paleo_df.columns:
			if col != 'year':
				paleo_df[col + '_shifted_forward'] = paleo_df[col].shift(-1)
				paleo_df[col + '_shifted_backward'] = paleo_df[col].shift(1)
	if scale:
		scaler = MinMaxScaler()
		columns = [col for col in paleo_df.columns if col not in ['year']]
		paleo_df[columns] = scaler.fit_transform(paleo_df[columns])
	
	if not return_metadata:
		return paleo_df
	else:
		valid_ids = paleo_df.columns.tolist()
		filtered_metadata = filtered_metadata[filtered_metadata['paleoData_TSid'].isin(valid_ids)]
		return paleo_df, filtered_metadata


def prepare_prcp_df(fname, scale=True):
	
	if "aismr" in fname:
		prcp_df = pd.read_csv("../monsoon/aismr.csv", sep='\t')
	
	elif fname[0] in '1234567':
		prcp_df = pd.read_csv(f"../data/{fname}.txt", delim_whitespace=True, skiprows=2)
		prcp_df = prcp_df.rename(columns={"YEAR": "year", "JJAS": "summer_prcp"})
	
	else:
		prcp_df = pd.read_csv(f"../monsoon/{fname}.csv")
	

	if scale:
		scaler = StandardScaler()
		columns = [col for col in prcp_df.columns if col not in ['year']]
		prcp_df[columns] = scaler.fit_transform(prcp_df[columns])
	
	return prcp_df


def combined_loss(alpha=1.0, beta=1.0, gamma=1.0):
	from tensorflow.keras import backend as K
	import tensorflow as tf
	def loss(y_true, y_pred):
		# Flattening tensors
		y_true_flat = K.flatten(y_true)
		y_pred_flat = K.flatten(y_pred)

		# Masking tensors to keep non-zero values of y_true
		mask = K.cast(K.not_equal(y_true_flat, 0), dtype=tf.float32)
		y_true_masked = y_true_flat * mask
		y_pred_masked = y_pred_flat * mask

		# 1. Mean Squared Error
		mse = K.mean(K.square(y_pred - y_true))

		# 2. Variance Difference
		true_variance = K.var(y_true_masked)
		pred_variance = K.var(y_pred_masked)
		var_diff = K.abs(true_variance - pred_variance)

		# 3. Pattern Correlation Coefficient		
		# Computing Pearson correlation
		cov = K.mean(y_true_masked * y_pred_masked) - K.mean(y_true_masked) * K.mean(y_pred_masked)
		std_y_true = K.std(y_true_masked)
		std_y_pred = K.std(y_pred_masked)
		correlation = cov / (std_y_true * std_y_pred + K.epsilon())  # adding epsilon to prevent division by zero

		# Since we want to maximize correlation (and correlation is between -1 and 1),
		# we'll convert it into a loss to minimize: 1 - correlation
		corr_loss = 1 - correlation

		return alpha * mse + beta * var_diff + gamma * corr_loss

	return loss


def blended_loss(alpha=0.5):
	from tensorflow.keras import backend as K
	"""
	Blends MSE and variance-based loss.
	
	Parameters:
	- alpha: weighting factor between 0 and 1. 
			 alpha = 1 means only MSE, alpha = 0 means only variance loss.

	Returns:
	- Custom loss function
	"""
	def loss(y_true, y_pred):
		mse_loss = mean_squared_error(y_true, y_pred)
		
		variance_true = K.var(y_true)
		variance_pred = K.var(y_pred)
		variance_loss = K.square(variance_true - variance_pred)
		
		return alpha * mse_loss + (1 - alpha) * variance_loss

	return loss


def kge_loss(y_true, y_pred):
	from tensorflow.keras import backend as K
	
	# Pearson correlation coefficient
	r = K.mean((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred))) / (0.0001 + K.std(y_true) * K.std(y_pred))

	# Standard deviation ratio
	alpha = K.std(y_pred) / (0.0001 + K.std(y_true))
	
	# Bias ratio
	beta = K.mean(y_pred) / (0.0001 + K.mean(y_true))
	
	# KGE
	kge = 1 - K.sqrt(K.square(r - 1) + K.square(alpha - 1) + K.square(beta - 1))

	return K.log(1-kge)  # We negate it because Keras minimizes the loss and we want to maximize KGE


# Compute Kling-Gupta Efficiency
def compute_kge(y_true, y_pred):
	sd_sim = np.std(y_pred)
	sd_obs = np.std(y_true)
	mean_sim = np.mean(y_pred)
	mean_obs = np.mean(y_true)
	r = np.corrcoef(y_true, y_pred)[0, 1]
	kge = 1 - np.sqrt((r - 1)**2 + (sd_sim / sd_obs - 1)**2 + (mean_sim / mean_obs - 1)**2)
	return kge



def build_model(X_train, loss_func):
	import tensorflow as tf

	from tensorflow.keras.models import Sequential
	from keras.layers import Dense, Dropout, Reshape, UpSampling2D, Conv2D, Cropping2D
	from tensorflow.keras.optimizers import Adam
	from tensorflow.keras import backend as K
	from tensorflow.keras.callbacks import EarlyStopping
	from tensorflow.keras.losses import mean_squared_error

	from tensorflow.keras.layers import BatchNormalization
	from tensorflow.keras.regularizers import l1, l2


	model = Sequential([
		# Dense layers to process the tabular data
		tf.keras.layers.Dense(512, activation='linear', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(256, activation='relu',),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Dense(64*32*32, activation='relu'),  # This prepares data for a 32x32 low-res version.
		
		# Reshape into a 'low-resolution' spatial format
		tf.keras.layers.Reshape((32, 32, 64)),
		
		# Upscale using Conv2DTranspose layers
		tf.keras.layers.Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', activation='relu'),
		tf.keras.layers.Conv2DTranspose(32, (3,3), strides=(3,3), padding='same', activation='relu'),
		tf.keras.layers.Conv2DTranspose(1, (3,3), padding='same', activation='linear'),  
		tf.keras.layers.Cropping2D(cropping=((0, 63), (0, 57))) 
	])
	
	model.compile(optimizer=Adam(), loss=loss_func)
	
	return model


def build_conv_model(X_train, loss_func):

	import tensorflow as tf
	physical_devices = tf.config.list_physical_devices('GPU')

	from tensorflow.keras.models import Sequential
	from keras.layers import Dense, Dropout, Reshape, UpSampling2D, Conv2D, Cropping2D
	from tensorflow.keras.optimizers import Adam
	from tensorflow.keras import backend as K
	from tensorflow.keras.callbacks import EarlyStopping
	from tensorflow.keras.losses import mean_squared_error

	from tensorflow.keras.layers import BatchNormalization
	from tensorflow.keras.regularizers import l1, l2


	model = Sequential([
		# Dense layers to process the tabular data
		Dense(512, activation='linear', kernel_regularizer=l2(0.01), input_shape=(X_train.shape[1],)),
		Dropout(0.2),
		Dense(256, activation='relu'),
		Dropout(0.2),
		Dense(64*32*32, activation='relu'),  # This prepares data for a 32x32 low-res version.

		# Reshape into a 'low-resolution' spatial format
		Reshape((32, 32, 64)),

		# Upscale using Conv2D and UpSampling2D layers
		UpSampling2D((2, 2)),  # Double the spatial dimensions (i.e., 64x64)
		Conv2D(64, (3,3), padding='same', activation='relu'),

		UpSampling2D((3, 3)),  # Triple the spatial dimensions (i.e., 192x192)
		Conv2D(32, (3,3), padding='same', activation='relu'),

		Conv2D(1, (3,3), padding='same', activation='linear'),  
		Cropping2D(cropping=((0, 63), (0, 57)))
	])

	model.compile(optimizer=Adam(), loss=loss_func)
	
	return model



def split_train_val_test(paleo_df, prcp_data, prcp_years, train_years, val_years, test_years, verbose=0):

	y_train = prcp_data[np.in1d(prcp_years, train_years)]
	y_test = prcp_data[np.in1d(prcp_years, test_years)]
	y_val = prcp_data[np.in1d(prcp_years, val_years)]

	X_train = paleo_df[paleo_df['year'].isin(train_years)].drop(columns='year').values[:, 1:]
	X_test = paleo_df[paleo_df['year'].isin(test_years)].drop(columns='year').values[:, 1:]
	X_val = paleo_df[paleo_df['year'].isin(val_years)].drop(columns='year').values[:, 1:]

	X_extended = paleo_df[(paleo_df['year'] >= 1500) & (paleo_df['year'] < 1901)].drop(columns='year').values[:, 1:]

	if verbose:
		print("X_train shape:", X_train.shape)
		print("y_train shape:", y_train.shape)
		print("X_test shape:", X_test.shape)
		print("y_test shape:", y_test.shape)

		if np.isnan(X_train).sum() == 0:
			print("X_train has no NaN values.")
		else:
			print("X_train has NaN values.")

		if np.isnan(X_test).sum() == 0:
			print("X_test has no NaN values.")
		else:
			print("X_test has NaN values.")

	return X_train, X_val, X_test, X_extended, y_train, y_test, y_val


def plot_training_loss(history):
	plt.plot(history.history['loss'], label='train loss')
	plt.plot(history.history['val_loss'], label='validation loss')
	plt.xlabel('Epochs')
	plt.ylabel('MSE Loss')
	plt.legend()
	plt.show()


def get_pcc_and_vd(y_test, y_pred):
	test_data = y_test[0].squeeze()
	predicted_data = y_pred[0].squeeze()

	test_data_flat = test_data.flatten()
	predicted_data_flat = predicted_data.flatten()

	# Mask the arrays to consider only non-zero observed data
	non_zero_mask = test_data_flat != 0
	test_data_masked = test_data_flat[non_zero_mask]
	predicted_data_masked = predicted_data_flat[non_zero_mask]

	# Compute correlation coefficient
	corr_coeff_matrix = np.corrcoef(test_data_masked, predicted_data_masked)
	pattern_corr_coeff = corr_coeff_matrix[0, 1]
	
	variance_match = np.abs(np.var(test_data_masked)-np.var(predicted_data_masked))

	return pattern_corr_coeff, variance_match

from shapely.geometry import Polygon
from shapely.ops import unary_union
import cartopy.crs as ccrs

def plot_homogeneous_regions(ax, gdf, plot=True):
	
	gdf = gdf[gdf['geometry'].notnull()]
	gdf = gdf[~gdf["ST_NM"].isin(["Lakshadweep", "Andaman & Nicobar Island"])]
	
	india_boundary = gdf.unary_union

	# Define the polygons based on bounds
	NMI_states = ["Jammu & Kashmir", "Ladakh", "Himachal Pradesh", "Uttarakhand"]
	NEI_states = ["Assam", "Manipur", "Meghalaya", "Mizoram", "Tripura", "West Bengal"]

	NMI_additional = Polygon([(75.87, 32.63), (77.13, 32.63), (77.13, 33.64), (75.87, 33.64)])
	NMI = gdf[gdf["ST_NM"].isin(NMI_states)].geometry.unary_union.union(NMI_additional)
	
	NEI_additional = Polygon([(87, 14.5), (100, 14.5), (100, 50), (87, 50)])
	NEI = gdf[gdf["ST_NM"].isin(NEI_states)].geometry.unary_union.union(NEI_additional)

	NWI = Polygon([(40, 20.5), (80, 20.5), (80, 50), (40, 50)]).difference(NMI)
	NCI = Polygon([(80, 20.5), (87, 20.5), (87, 30), (80, 30)])
	WPI = Polygon([(40, 14.5), (79, 14.5), (79, 20.5), (40, 20.5)])
	EPI = Polygon([(79, 14.5), (96, 14.5), (96, 20.5), (79, 20.5)])
	SPI = Polygon([(40, 0), (96, 0), (96, 14.5), (40, 14.5)])

	NWI = NWI.difference(NMI)
	NCI = NCI.difference(NMI)
	NEI = NEI.difference(NCI)

	regions = {
	"NMI": NMI.intersection(india_boundary),
	"NWI": NWI.intersection(india_boundary),
	"NCI": NCI.intersection(india_boundary),
	"NEI": NEI.intersection(india_boundary),
	"WPI": WPI.intersection(india_boundary),
	"EPI": EPI.intersection(india_boundary),
	"SPI": SPI.intersection(india_boundary)
	}

	label_offsets = {
		"NMI": (0, 1),  # offset by 0 degrees in longitude, 1 degree in latitude
		"NEI": (0, 0.5)  # offset by 0 degrees in longitude, 0.5 degrees in latitude
	}

	if plot == True:
		# Plot each region with its label
		for name, region in regions.items():
			ax.add_geometries([region], crs=ccrs.PlateCarree(), facecolor='none', edgecolor='red', linewidth=1.5, zorder=10)
			
			label_point = region.representative_point()
			# Apply the label offsets if they exist for the given region
			offset = label_offsets.get(name, (0, 0))
			ax.text(label_point.x + offset[0], label_point.y + offset[1], name, horizontalalignment='center', verticalalignment='center', fontsize=12, color='red', transform=ccrs.PlateCarree())

		return ax
	
	else:
		return regions

def fetch_cnn_ensemble(source='IMD', pcc_thresh=0.3, vd_thresh=0.2, return_all_models = False):

	find = lambda x, arr: np.argmin(np.abs(x-arr))
	lats = np.arange(6.5, 38.75, 0.25)
	lons = np.arange(66.5, 100.25, 0.25)

	prcp_data = np.load(f"../monsoon/{source}-JJAS-means.npy")
	prcp_mean = prcp_data.mean(axis=0)
	prcp_std = prcp_data.std(axis=0)

	df = pd.read_csv(f"ensemble-cnn/{source}-results.csv")

	filtered_df = df[(df['pcc'] > pcc_thresh) & (df['vd'] < vd_thresh)]
	good_models = filtered_df['model_code'].tolist()

	#print(good_models)
	print(len(good_models))

	year_list = np.arange(1500,1995)


	ensemble = []

	for model_code in good_models:
		
		output = np.load(f"ensemble-cnn/output/{model_code}.npy").squeeze()	
		ensemble.append(output)
		print(model_code, end="\r")
		
	if not return_all_models:
		ensemble_mean = np.mean(ensemble, axis=0)
		return ensemble_mean, lons, lats, year_list
	
	else:
		return np.array(ensemble), lons, lats, year_list


def make_colormap(colors):
	return mcolors.LinearSegmentedColormap.from_list("", colors)
   
# Define the individual colormaps
red_cmap = make_colormap([(1, 0.8, 0.8), (0.8, 0, 0)])
yellow_cmap = make_colormap([(1, 1, 0.5), (1, 0.65, 0)])
white_cmap = make_colormap([(1, 1, 1), (1, 1, 1)])
green_cmap = make_colormap([(0.5, 1, 0.5), (0, 0.5, 0)])
blue_cmap = make_colormap([(0.5, 0.5, 1), (0, 0, 1)])

# Now, combine them
n = 32
newcolors = np.vstack((
	yellow_cmap(np.linspace(1, 0, n)),
	red_cmap(np.linspace(1, 0, n)),
	white_cmap(np.linspace(0, 1, n)),
	blue_cmap(np.linspace(0, 1, n)),
	green_cmap(np.linspace(0, 1, n))
))
newcmp = mcolors.ListedColormap(newcolors)
