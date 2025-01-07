import lipd
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt

data_iso2k   = lipd.readLipd('/home/users/rz908899/mitre/paleo/iso2k/')
data_iso2k_ts = lipd.extractTs(data_iso2k)

data_iso2k_ts_filtered = lipd.filterTs(data_iso2k_ts,'maxYear > 1950')
data_iso2k_ts_filtered = lipd.filterTs(data_iso2k_ts_filtered,'geo_meanLat > 30')
data_iso2k_ts_filtered = lipd.filterTs(data_iso2k_ts_filtered,'geo_meanLat < 70')
data_iso2k_ts_filtered = lipd.filterTs(data_iso2k_ts_filtered,'geo_meanLon > -20')
data_iso2k_ts_filtered = lipd.filterTs(data_iso2k_ts_filtered,'geo_meanLon < 20')


def create_series_from_dict(d):
    column_name = d['dataSetName']
    
    # Round years to the nearest integer and then convert
    rounded_years = np.round(np.array(d['year']).astype(float)).astype(int)
    
    # Use dict to handle data
    year_value_dict = dict(zip(rounded_years, d['paleoData_values']))
    return pd.Series(year_value_dict, name=column_name)


# Create the main DataFrame with data
series_list = [create_series_from_dict(d) for d in data_iso2k_ts_filtered if 'year' in d.keys()]
df_data = pd.concat(series_list, axis=1)
df_data = df_data.reindex(index=range(1000, 2021)).sort_index()


# Extracting metadata
metadata_keys = ['mode', 'time_id', 'archiveType', 'geo_meanLon', 'geo_meanLat', 'geo_meanElev', 'geo_type', 'collectionName', 'createdBy', 'dataSetName', 'hasPaleoDepth', 'investigators', 'tagMD5', 'lipdverseLink', 'maxYear', 'minYear', 'nUniqueAges', 'archiveTypeOriginal', 'hasDepth', 'hasChron', 'originalDataUrl', 'datasetId', 'changelog', 'pub1_author', 'pub1_abstract', 'pub1_journal', 'pub1_iso', 'pub1_publisher', 'pub1_title', 'pub1_type', 'pub1_doi', 'pub1_year', 'pub1_citation', 
 'geo_country', 'geo_location', 'geo_siteName', '@context', 'lipdVersion', 'tableType',  'paleoData_tableName', 'paleoData_missingValue', 'paleoData_filename']

df_metadata = pd.DataFrame([{k: d[k] for k in metadata_keys if k in d} for d in data_iso2k_ts_filtered])
df_metadata = df_metadata.set_index('dataSetName')

# Define the range for which you need complete data
complete_range = range(1940, 1980)

# Find columns that have NaNs in the specified range and drop them
cols_to_drop = df_data.loc[complete_range].columns[df_data.loc[complete_range].isna().any()]

df_data = df_data.drop(columns=cols_to_drop)
df_metadata = df_metadata.drop(index=cols_to_drop, errors='ignore')

print(df_data)
print(df_metadata)

df_data.to_csv("/home/users/rz908899/mitre/paleo/cet/iso2k_processed.csv")
df_metadata.to_csv("/home/users/rz908899/mitre/paleo/cet/iso2k_processed_metadata.csv")


# Extract the necessary information from the metadata dataframe
lats = df_metadata['geo_meanLat'].astype(float).values
lons = df_metadata['geo_meanLon'].astype(float).values
types = df_metadata['archiveType'].values

# Get a set of unique types and assign each a unique color
unique_types = list(set(types))
colors = plt.cm.jet(np.linspace(0, 1, len(unique_types)))
type_color_map = dict(zip(unique_types, colors))

# Create the map plot
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(15, 10))
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Plot each dataset's location
for lat, lon, archive_type in zip(lats, lons, types):
    ax.plot(lon, lat, marker='o', color=type_color_map[archive_type], markersize=5, label=archive_type)

# Add gridlines and labels
gl = ax.gridlines(draw_labels=True, linewidth=1, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False

# Create a custom legend without repeating labels
handles, labels = ax.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.show()

