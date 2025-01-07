import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functions import *
from scipy.stats import pearsonr

kge_thresh = 0.2
r_thresh = 0.5

fnames = ['1-nmi', '2-nwi', '3-nci', '4-nei', '5-wpi', '6-epi', '7-spi', 'aismr']

famine_df = pd.read_csv("../data/cleaned_famines.csv", encoding='latin1')

fig, axes = plt.subplots(len(fnames), 1, figsize=(14, 35), sharey=True, gridspec_kw={'hspace': 0})  # 7 rows, 1 column of subplots


for idx, fname in enumerate(fnames):
	ax = axes[idx]
	
	metadata_df = pd.read_csv(f"ensemble/metadata-{fname}.csv")
	
	# Filter models based on the thresholds
	filtered_models = metadata_df[(metadata_df['KGE'] > kge_thresh) & 
								 (metadata_df['linear_correlation_coefficient'] > r_thresh)]['model_label'].tolist()
	print(fname, len(filtered_models))
	
	if not filtered_models:
		filtered_models = metadata_df[metadata_df['linear_correlation_coefficient'] > 0].nlargest(10, 'KGE')['model_label'].tolist()
		print(metadata_df[metadata_df['KGE'] > 0].nlargest(10, 'linear_correlation_coefficient')['linear_correlation_coefficient'])

	results_df = pd.read_csv(f"ensemble/results-{fname}.csv")
	columns_to_include = ['year'] + [col for col in results_df.columns if any(model in col for model in filtered_models)]
	results_df = results_df[columns_to_include]

	prcp_df = prepare_prcp_df(fname)
	prcp_years = prcp_df['year'].values
	prcp_values = prcp_df['summer_prcp'].values
	
	train_columns = [col for col in results_df.columns if col.split("_")[-1] == 'train']
	test_columns = [col for col in results_df.columns if col.split("_")[-1] in ('test', 'extended')]
	test_median = results_df[test_columns].median(axis=1)
	test_upper = results_df[test_columns].quantile(0.95, axis=1)
	test_lower = results_df[test_columns].quantile(0.05, axis=1) 
	
	ax.fill_between(results_df['year'], test_lower, test_upper, color='red', alpha=0.2, label="Ensemble spread")
	ax.plot(results_df['year'], test_median, 'k-', label="Ensemble median" , lw=0.5)
	ax.plot(prcp_years, prcp_values, 'g-', label='Observed', lw=0.5)
	
	# Extracting the year range and values of test_median
	years = results_df['year'].values
	test_median_values = test_median.values

	valid_test = ~np.isnan(test_median) & np.in1d(years, prcp_years)
	valid_obs = np.in1d(prcp_years, years[valid_test])
	
	print(np.sum(valid_test))
	print(np.sum(valid_obs))
	
	r_value, _ = pearsonr(test_median[valid_test], prcp_df.loc[valid_obs, 'summer_prcp'])

	for _, row in famine_df.iterrows():
		# Check if the current region (fname) had the famine (value in its column == 1)
		if row[fname] == 1:
			# Extract famine start and end year
			start_year = row['Start Year'] - 1
			end_year = row['End Year']

			# Check if there's any year within the famine range where test_median falls below -0.5
			mask = (years >= start_year) & (years <= end_year)
			mask_obs = (prcp_years >= start_year) & (prcp_years <= end_year)
			if any(test_median_values[mask] < -1) or any(prcp_values[mask_obs] < -1):
				ax.axvspan(start_year, end_year, color='grey', alpha=0.33)
	
	
	ax.set_ylim(-3.5, 3.5)
	ax.set_xlim(1500,1995)
	
	ax.set_xticks(np.arange(1500,2000,100), minor=False)
	ax.set_xticks(np.arange(1500,2000,25), minor=True)

	# Remove x-axis labels for all but the bottom plot
	if idx < len(fnames) - 1:
		ax.set_xticklabels([])

	# Set the zorder of axes to ensure ticks from upper subplots are visible
	ax.set_zorder(len(fnames) - idx)  # give higher zorder to upper plots
	ax.patch.set_visible(False)
	

	ax.set_yticks([-2, -1, 0, 1, 2])
		
	# Draw faint horizontal line at y=0
	ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')

	# Capitalise and format the subplot titles
	label = fname.upper().split('-')[1] if fname != 'aismr' else 'AI'
	ax.text(0.99, 0.025, f"{label} (r = {r_value:.3f})", transform=ax.transAxes, ha="right", va="bottom", fontsize=12, fontweight="bold")


	ax.set_zorder(len(fnames) - idx)  # give higher zorder to upper plots
	ax.patch.set_visible(False)


ax.set_xlabel('Year')
fig.text(0.06, 0.5, 'Standardised summer precipitation', ha='center', va='center', rotation='vertical')
# Set legend
import matplotlib.patches as mpatches
handles, labels = ax.get_legend_handles_labels()
famine_patch = mpatches.Patch(color='grey', alpha=0.33, label='Famine')
handles.append(famine_patch)
fig.legend(handles, labels + ['Famine'], loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=4)

#plt.tight_layout()
plt.show()

