import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import matplotlib.colors as mcolors
from functions import *
from scipy.ndimage import generic_filter
from scipy.stats import linregress
from glob import glob

def median_filter(arr):
	smoothed_arr = generic_filter(arr, np.median, size=(1, 9, 9), mode='constant', cval=np.NaN, origin=0)
	return(smoothed_arr)


def compute_correlations(model_years, model_rains):
	model_r = []
	for year, rain in zip(model_years, model_rains):
		
		if year not in prcp_years: continue
		idt = find(year, prcp_years)
		obs_rainfall = prcp_data[idt]
		smoothed_obs = generic_filter(obs_rainfall.squeeze(), np.median, size=(9, 9), mode='constant', cval=np.NaN, origin=0)
		mask = ~np.isnan(smoothed_obs)
		smoothed_model = generic_filter(rain.squeeze(), np.median, size=(9, 9), mode='constant', cval=np.NaN, origin=0)
		_,_,r,_,_ = linregress(smoothed_obs[mask].ravel(), smoothed_model[mask].ravel())
		model_r.append(r)
		print(year,r)

	return(np.array(model_r))


def get_paleo_model(source):
	df = pd.read_csv(f"ensemble-cnn/{source}-results.csv")
	filtered_df = df[(df['pcc'] > 0.3) & (df['vd'] < 0.2)]
	good_models = filtered_df['model_code'].tolist()

	start_year = 1901
	end_year = 1994
	length = end_year-start_year+1

	test_year_range = np.arange(start_year,end_year+1)
	full_year_range = np.arange(1500,1995+1)
	
	predicted = []
	predicted_years = []
	
	
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
			predicted_years.append(year)

	predicted = np.array(predicted)
	predicted_years = np.array(predicted_years)
	
	mean_years = np.unique(predicted_years)
	mean_predicted = []
	
	for y in mean_years:
		mean_predicted.append(np.mean(predicted[predicted_years==y], axis=0))
	
	return(mean_years, mean_predicted)


def get_regression_model(source):
	
	flist = glob(f"{source}/*.npy")
	years = []
	predicted = []
	
	for f in flist:
		rainfall = np.load(f).squeeze()
		if rainfall.size==1:
			continue
		year = int(f.split("/")[-1].split(".")[0])
		if year<1900:
			continue
		years.append(year)
		predicted.append(rainfall)
	
	
	return(np.array(years), np.array(predicted))
	



find = lambda x, arr: np.argmin(np.abs(x-arr))
lats = np.arange(6.5, 38.75, 0.25)
lons = np.arange(66.5, 100.25, 0.25)

prcp_years, prcp_data = load_gridded_prcp(source='IMD')
era_years, era_data = load_gridded_prcp(source='ERA5')

era_r = compute_correlations(era_years, era_data)

cnn_years, cnn_data = get_paleo_model("IMD")
cnn_r = compute_correlations(cnn_years, cnn_data)

cnne_years, cnne_data = get_paleo_model("ERA5")
cnne_r = compute_correlations(cnne_years, cnne_data)

mvlr_years, mvlr_data = get_regression_model("mvlr")
mvlr_r = compute_correlations(mvlr_years, mvlr_data)

eof_years, eof_data = get_regression_model("eof")
eof_r = compute_correlations(eof_years, eof_data)

colors = ['r', 'orange', 'y', 'g', 'b']

plt.figure(figsize=(10, 6))
data_to_plot = [era_r, cnn_r, cnne_r, mvlr_r, eof_r]

for n, (c, data) in enumerate(zip(colors, data_to_plot)):
	plt.scatter(np.random.normal(n+1, 0.04, len(data)), data, color=c, alpha=0.4)

plt.boxplot(data_to_plot, vert=True, whis=(5,95), 
            medianprops = {'color':'k', 'lw':1}, showfliers=False,
            labels=['ERA5', 'CNN model\n(IMD)', 'CNN model\n(ERA5)', 'multivariate\nlinear regression', 'PCA regression'])
plt.ylabel('PCC with IMD observations')
plt.grid(True, axis='y', lw=0.5)
plt.axhline(0, color='darkgrey',lw=1.5)
plt.show()





