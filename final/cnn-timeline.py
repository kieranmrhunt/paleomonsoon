from functions import *
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import colormaps as cmaps
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
import pandas as pd

# Load ENSO data
enso_data = pd.read_csv("../data/enso-reconstructed.dat", sep="\s+")
enso_years = enso_data['age_AD']
enso_sst_anom = enso_data['sst.anom']

# Load eruption data
eruption_data = pd.read_csv("../data/eruptions.csv")

# Extract relevant columns for VEI
eruption_data['Start Year'] = pd.to_numeric(eruption_data['Start Year'], errors='coerce')
eruption_data['VEI'] = pd.to_numeric(eruption_data['VEI'], errors='coerce')

# Filter data where VEI is 4 or over
filtered_eruption_data = eruption_data[eruption_data['VEI'] >= 4]

# Group by 'Start Year' and get the maximum VEI for each year
max_vei_per_year = filtered_eruption_data.groupby('Start Year')['VEI'].max()

# Main precipitation data processing
cnn_ensemble_prcp, lons, lats, cnn_year_list = fetch_cnn_ensemble('IMD', return_all_models=True)
cnn_ensemble_mean_prcp = np.mean(cnn_ensemble_prcp, axis=0)
year_mask = cnn_year_list <= 1900
cnn_ensemble_mean_prcp = cnn_ensemble_mean_prcp[year_mask]
cnn_year_list = cnn_year_list[year_mask]

imd_year_list, imd_prcp = load_gridded_prcp('IMD')
mask = ~np.isnan(imd_prcp.mean(axis=(0, 3)))[None, :, :]
imd_prcp = imd_prcp.squeeze()

cnn_prcp_timeseries = standardise(np.nanmean(destandardise(cnn_ensemble_mean_prcp), axis=(1, 2)))
cnn_prcp_all_models = [standardise(np.nanmean(destandardise(model_prcp), axis=(1, 2))) for model_prcp in cnn_ensemble_prcp]
cnn_upper_bound = np.nanmax(cnn_prcp_all_models, axis=0)[year_mask]
cnn_lower_bound = np.nanmin(cnn_prcp_all_models, axis=0)[year_mask]

imd_prcp_timeseries = standardise(np.nanmean(destandardise(imd_prcp), axis=(1, 2)))

# Combine all year data
all_years = np.concatenate([cnn_year_list, imd_year_list])
all_prcp = np.concatenate([cnn_prcp_timeseries, imd_prcp_timeseries])

# Low-pass filtered rainfall (Gaussian smoothed)
lowpass_rainfall = gaussian_filter(all_prcp, 10)

# ENSO data normalisation
enso_mean = np.mean(enso_sst_anom[enso_years <= 1500])
enso_std = np.std(enso_sst_anom[enso_years <= 1500])
enso_sst_anom = (enso_sst_anom - enso_mean) / enso_std

# Decadally smoothed ENSO
decadal_smoothed_enso = gaussian_filter(enso_sst_anom, sigma=10)

# Prepare VEI data with matching years
vei_years = max_vei_per_year.index.values
vei_values = max_vei_per_year.values

# Create a dataframe for all data
data_dict = {
    'Year': all_years,
    'Ensemble Mean Rainfall': np.concatenate([cnn_prcp_timeseries, imd_prcp_timeseries]),
    'Decadally Smoothed Rainfall': lowpass_rainfall,
    'ENSO': np.interp(all_years, enso_years, enso_sst_anom),
    'Decadally Smoothed ENSO': np.interp(all_years, enso_years, decadal_smoothed_enso),
    'Maximum VEI': np.interp(all_years, vei_years, vei_values, left=np.nan, right=np.nan)
}

# Convert dictionary to DataFrame
output_df = pd.DataFrame(data_dict)

# Save to CSV
output_df.to_csv('output_climate_data.csv', index=False)

# Create a figure with GridSpec for precise layout
fig = plt.figure(figsize=(10, 6.5))
gs = GridSpec(3, 1, height_ratios=[3, 2, 1], hspace=0)  # Three rows, no vertical space between

# Create first axis (ax1) for precipitation data
ax1 = fig.add_subplot(gs[0])

# Precipitation data
rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))
ax1.bar(all_years, all_prcp, color=cmaps.BlueYellowRed_r(rescale(cnn_prcp_timeseries)), label='Precipitation')
ax1.fill_between(cnn_year_list, cnn_lower_bound, cnn_upper_bound, color='lightgrey', zorder=0)

# Plot smoothed rainfall data
ax1.plot(all_years, lowpass_rainfall * 3, color='k', ls=':', lw=0.5)

ax1.axvline(1900, color='g', ls='--')
ax1.set_xlim([1500, 2023])
ax1.set_ylabel("(a) All-India seasonal rainfall\n(standardised anomaly)")

# Add gridlines for centuries and 25 years (only on x-axis)
ax1.grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.75)
ax1.grid(which='minor', axis='x', color='lightgray', linewidth=0.5)
ax1.set_xticks(np.arange(1500, 2100, 100))  # Major ticks every century
ax1.set_xticks(np.arange(1500, 2100, 25), minor=True)  # Minor ticks every 25 years

# Create second axis (ax2) for ENSO data (new axis between rainfall and VEI)
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# Plot ENSO data
ax2.plot(enso_years, enso_sst_anom, color='green', lw=1, label='ENSO SST Anomaly')
ax2.set_ylabel('(b) ENSO\n(stand. anom.)', color='green')

# Plot decadal smoothed ENSO
ax2.plot(enso_years, decadal_smoothed_enso * 3, color='k', lw=0.5, ls=':', label='Decadal Smoothed ENSO')

# Add gridlines for centuries and 25 years (only on x-axis)
ax2.grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.75)
ax2.grid(which='minor', axis='x', color='lightgray', linewidth=0.5)

ax2.set_yticks([-3, -2, -1, 0, 1, 2, 3])

# Create third axis (ax3) for VEI data
ax3 = fig.add_subplot(gs[2], sharex=ax1)

# Plot VEI as red bars
ax3.bar(vei_years, vei_values, color='red', alpha=0.7, width=2, label='Maximum VEI')

# Set labels and scale for ax3 (no log scale, actual VEI)
ax3.set_ylabel('(c) VEI\n(annual max.)', color='red')
ax3.set_ylim(4, 7)

# Add gridlines for ax3 (only on x-axis)
ax3.grid(which='major', axis='x', color='gray', linestyle='-', linewidth=0.75)
ax3.grid(which='minor', axis='x', color='lightgray', linewidth=0.5)

ax1.set_xlim([1500, 2023])

# Hide x-axis labels for ax1 and ax2
plt.setp(ax1.get_xticklabels(), visible=False)
plt.setp(ax2.get_xticklabels(), visible=False)

plt.show()

