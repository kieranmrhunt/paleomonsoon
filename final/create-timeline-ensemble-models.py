from functions import *

set_seed()

# Define region and resolution
x1, x2 = 40, 120  
y1, y2 = -10, 50
min_resolution = 10
paleo_df = prepare_paleo_df(x1,x2,y1,y2,min_resolution)

print(paleo_df)


def generate_model_label(config, period, fname):
    f = fname.split("_")[-1]
    return f"{f}_L1_{config['L1_neurons']}_L2_{config['L2_neurons']}_dropout_{config['dropout']}_testPeriod_{period[0]}_{period[1]}"


def build_model(input_shape, config):
	model = Sequential([
		Dense(config['L1_neurons'], activation=config['L1_activation'], 
			  input_shape=input_shape, kernel_regularizer=l1(config['L1_regularisation'])), 
		Dropout(config['dropout']), 
		Dense(config['L2_neurons'], activation='linear'), 
		Dense(1)
	])
	loss_func = combined_loss(alpha=1., beta=1, gamma=2.)
	model.compile(optimizer=Adam(), loss=loss_func)
	return model


# Set up ensemble configurations and periods
test_periods = [(y,y+10) for y in range(1902,1985,2)]
configs = [
	{'L1_neurons': 100, 'L2_neurons': 20, 'L1_activation': 'relu', 'dropout': 0.2, 'L1_regularisation': 0.01},
	{'L1_neurons': 100, 'L2_neurons': 20, 'L1_activation': 'relu', 'dropout': 0.0, 'L1_regularisation': 0.05},
	{'L1_neurons': 100, 'L2_neurons': 20, 'L1_activation': 'linear', 'dropout': 0.2, 'L1_regularisation': 0.01},
	{'L1_neurons': 20,  'L2_neurons': 5,  'L1_activation': 'relu', 'dropout': 0.1, 'L1_regularisation': 0.01},
	{'L1_neurons': 100, 'L2_neurons': 10, 'L1_activation': 'linear', 'dropout': 0.2, 'L1_regularisation': 0.001},
	{'L1_neurons': 50, 'L2_neurons': 5, 'L1_activation': 'linear', 'dropout': 0.2, 'L1_regularisation': 0.01},
	{'L1_neurons': 50, 'L2_neurons': 5, 'L1_activation': 'relu', 'dropout': 0.2, 'L1_regularisation': 0.01},
]

#fnames = ['1-nmi', '2-nwi', '3-nci', '4-nei', '5-wpi', '6-epi', '7-spi', 'aismr', 'mean_seasonal_prcp_ERA5', 'mean_seasonal_prcp_IMD']

fnames = ['7-spi', 'aismr', 'mean_seasonal_prcp_ERA5', 'mean_seasonal_prcp_IMD']


ensemble_results = {
    'year': list(range(1500, 1996))
}

for fname in fnames:

	prcp_df = prepare_prcp_df(fname)
	results = []
	metadata_list = []

	for period in test_periods:
		for config in configs:
			K.clear_session()
			
			y0 = prcp_df['year'].min()
			y1 = 1995
			
			if period[0]<prcp_df['year'].min(): continue
			
			test_years = [y for y in range(period[0], period[1])]
			potential_valid_years = [y for y in range(y0, y1) if y not in test_years]
			valid_years = random.sample(potential_valid_years, 5)
			
			train_years = [y for y in range(y0,y1) if y not in test_years+valid_years]

			y_train = prcp_df[prcp_df['year'].isin(train_years)]['summer_prcp'].values
			y_test = prcp_df[prcp_df['year'].isin(test_years)]['summer_prcp'].values
			y_val = prcp_df[prcp_df['year'].isin(valid_years)]['summer_prcp'].values
			
			X_train = paleo_df[paleo_df['year'].isin(train_years)].drop(columns='year').values[:, 1:]
			X_test = paleo_df[paleo_df['year'].isin(test_years)].drop(columns='year').values[:, 1:]
			X_val = paleo_df[paleo_df['year'].isin(valid_years)].drop(columns='year').values[:, 1:]
			X_extended = paleo_df[(paleo_df['year'] >= 1500) & (paleo_df['year'] < y0)].drop(columns='year').values[:, 1:]
			
			

			model = build_model((X_train.shape[1],), config)
			early_stop = EarlyStopping(monitor='val_loss', patience=250, verbose=1, restore_best_weights=True)
			model.fit(X_train, y_train, epochs=3000, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=0)
			
			test_predictions = model.predict(X_test, verbose=0).ravel()
			train_predictions = model.predict(X_train, verbose=0).ravel()
			extended_predictions = model.predict(X_extended, verbose=0).ravel()

			slope, _, r_value, _, _ = linregress(test_predictions, y_test)
			kge = compute_kge(y_test, test_predictions)
			
			print(fname, config, period, kge, r_value)


			if kge > 0 and r_value > 0.3:
				results.append({
					'period': period,
					'config': config,
					'test_predictions': test_predictions,
					'train_predictions': train_predictions
				})
			else:
				continue
			
			
			model_label = generate_model_label(config, period, fname)

			
			metadata = {
				'model_label': model_label,
				'testing_period_start': test_years[0],
				'testing_period_end': test_years[-1],
				'L1_neurons': config['L1_neurons'],
				'L2_neurons': config['L2_neurons'],
				'L1_activation': config['L1_activation'],
				'dropout': config['dropout'],
				'L1_regularisation': config['L1_regularisation'],
				'KGE': kge,  
				'linear_correlation_coefficient': r_value 
			}

			
			full_length_train_predictions = np.full_like(ensemble_results['year'], np.nan, dtype=float)
			full_length_test_predictions = np.full_like(ensemble_results['year'], np.nan, dtype=float)
			full_length_extended_predictions = np.full_like(ensemble_results['year'], np.nan, dtype=float)

			train_years_array = prcp_df[prcp_df['year'].isin(train_years)]['year'].values
			test_years_array = prcp_df[prcp_df['year'].isin(test_years)]['year'].values
			extended_years_array = paleo_df[(paleo_df['year'] >= 1500) & (paleo_df['year'] < prcp_df['year'].min())]['year'].values

			full_length_train_predictions[np.isin(ensemble_results['year'], train_years_array)] = train_predictions.ravel()
			full_length_test_predictions[np.isin(ensemble_results['year'], test_years_array)] = test_predictions.ravel()
			full_length_extended_predictions[np.isin(ensemble_results['year'], extended_years_array)] = extended_predictions.ravel()

			ensemble_results[f'{model_label}_train'] = full_length_train_predictions
			ensemble_results[f'{model_label}_test'] = full_length_test_predictions
			ensemble_results[f'{model_label}_extended'] = full_length_extended_predictions

			metadata_list.append(metadata)



	# Convert the metadata list into a DataFrame and save as a CSV
	metadata_df = pd.DataFrame(metadata_list)
	metadata_df.to_csv(f"ensemble/metadata-{fname}.csv", index=False)

	# Convert the results list into a DataFrame and save as a CSV
	results_df = pd.DataFrame(ensemble_results)
	results_df.to_csv(f"ensemble/results-{fname}.csv", index=False)


'''
plt.figure(figsize=(14, 6))
plt.title('Ensemble Model Predictions vs. True Values')

# Extracting all the prediction columns (using train and test as an example)
train_columns = [col for col in results_df.columns if 'train' in col]
test_columns = [col for col in results_df.columns if 'test' in col]

# Compute the median and the spread of the predictions
train_median = results_df[train_columns].median(axis=1)
train_upper = results_df[train_columns].quantile(0.95, axis=1) # 95th percentile
train_lower = results_df[train_columns].quantile(0.05, axis=1) # 5th percentile

test_median = results_df[test_columns].median(axis=1)
test_upper = results_df[test_columns].quantile(0.95, axis=1)
test_lower = results_df[test_columns].quantile(0.05, axis=1)

# Plotting the median and the spread
plt.fill_between(results_df['year'], train_lower, train_upper, color='blue', alpha=0.2, label="Train 5th-95th percentile")
plt.plot(results_df['year'], train_median, 'b--', label="Train Median Prediction")

plt.fill_between(results_df['year'], test_lower, test_upper, color='red', alpha=0.2, label="Test 5th-95th percentile")
plt.plot(results_df['year'], test_median, 'r--', label="Test Median Prediction")

# True values
plt.plot(prcp_df['year'], prcp_df['summer_prcp'], 'k-', label='True Values')

plt.xlabel('Year')
plt.ylabel('Standardized Summer Precipitation')
plt.legend()
plt.show()
'''
