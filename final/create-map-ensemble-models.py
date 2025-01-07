from functions import *

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)
#tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

set_seed(1)

source = 'IMD'
prcp_years, prcp_data = load_gridded_prcp(source=source)


# Define region and resolution
#x1, x2 = 40, 95  
#y1, y2 = -10, 40
x1, x2 = 40, 120  
y1, y2 = -10, 50
min_resolution = 10
paleo_df = prepare_paleo_df(x1,x2,y1,y2,min_resolution)

print(paleo_df)

import sys; sys.exit()

#ensemble of 188 created by making a pair for all test years from 1901 to 1995
#val years +1 and +5.
#reject any models where PCC<0.4 and variance diff<0.1


start_year = prcp_years[0]
end_year = 1994
length = end_year-start_year+1

test_year_range = np.arange(start_year,end_year+1)
full_year_range = np.arange(1500,1995+1)

results = {'model_code':[], 'pcc':[], 'vd':[]}

for test_year in test_year_range:
	for j in [1,3,5,7]:
		
		model_code = f'{source}_{test_year}.{j}'
		print(model_code)
		
		test_years = [test_year,]
		val_years = np.array([(test_year-start_year+i+j)%length for i in range(0,length,20)])+start_year
		print(val_years)
		
		train_years = [y for y in range(start_year,1995) if (y not in test_years) and (y not in val_years)]

		X_train, X_val, X_test, X_extended, y_train, y_test, y_val = split_train_val_test(paleo_df, prcp_data, prcp_years, train_years, val_years, test_years)


		early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, restore_best_weights=True)
		loss_func = combined_loss(alpha=1., beta=0.75, gamma=0.25)


		model = build_model(X_train, loss_func)
		history = model.fit(X_train, y_train, epochs=1500, validation_data=(X_val, y_val), callbacks=[early_stop,], verbose=0)

		y_pred = model.predict(X_test, verbose=0)
		
		pattern_corr_coeff, variance_match = get_pcc_and_vd(y_test, y_pred)

		print(f"Pattern Correlation Coefficient: {pattern_corr_coeff:.3f}")
		print(f"Variance Difference: {variance_match:.3f}")
		
		results['model_code'].append(model_code)
		results['pcc'].append(pattern_corr_coeff)
		results['vd'].append(variance_match)
		
		y_extended = model.predict(X_extended, verbose=0)
		
		np.save(f'ensemble-cnn/output/{model_code}', y_extended)
		
		
		
df = pd.DataFrame(results)	
df.to_csv(f'ensemble-cnn/{source}-results.csv', index=False)		
		
		
		



