import lipd
import pandas as pd
import numpy as np
import glob
from scipy.interpolate import interp1d



def process_data(directory_path):

	basis_years = np.arange(0,2011)

	metadata_vars = ['paleoData_TSid','datasetName', 'archiveType', 'geo_meanLon', 'geo_meanLat', 'geo_siteName', 'pub1_dataUrl']


	df_metadata = pd.DataFrame(columns=metadata_vars + ['resolution',])
	df_values = pd.DataFrame(index=basis_years)

	files = glob.glob(directory_path+"*")
	

	for f in files:
		
		#if 'LS16THN301.lpd' not in f:
		#		continue

		ds_raw = lipd.readLipd(f)
		ds_list = lipd.extractTs(ds_raw)

		
		for ds in ds_list:

			if 'year' not in ds: continue
			
			try:
				years = np.array(ds['year']).astype(float)
				values = np.array(ds['paleoData_values']).astype(float)
			except ValueError as e:
				print(e)
				continue
			
			#print(ds)
			
			
			diffs = np.gradient(values)
			
			if np.sum(years>2020):
				years = 1950-years
			
			valid = ~(np.logical_or(np.isnan(years),np.isnan(values)))
			
			is_monotonically_increasing = np.all(diffs[valid] >= 0)
			is_monotonically_decreasing = np.all(diffs[valid] <= 0)
			
			years = years[valid]
			values = values[valid]
			
			if is_monotonically_increasing or is_monotonically_decreasing:
				continue
			
			if np.sum(years>1900)==0:
				continue
			
			
			#clean out duplicates before interpolation
			df = pd.DataFrame({'years': years, 'values': values})
			df = df.drop_duplicates(subset='years')
			years_cleaned = df['years'].values
			values_cleaned = df['values'].values
			
			if len(years_cleaned)<4: #i.e. not enough data to do cubic interpolation
				continue
			
			rebased_values = interp1d(years_cleaned, values_cleaned, kind='cubic',
				                      bounds_error=False)(basis_years)
			
			resolution = np.abs(np.mean(np.diff(years_cleaned)))
			
			
			metadata = {var: ds.get(var, np.nan) for var in metadata_vars}
			metadata['paleoData_TSid'] = ds.get('paleoData_TSid', np.nan)
			metadata['resolution'] = resolution
			
			df_metadata = df_metadata.append(metadata, ignore_index=True)
			df_values[ds['paleoData_TSid']] = rebased_values

			print(f, resolution)
			
	print(df_metadata)
	print(df_values)
	
	df_metadata.to_csv(directory_path + 'metadata.csv', index=False)
	df_values.to_csv(directory_path + 'values.csv', index=True)



process_data('/home/users/rz908899/mitre/paleo/iso2k/')
process_data('/home/users/rz908899/mitre/paleo/pages2k/')


	

