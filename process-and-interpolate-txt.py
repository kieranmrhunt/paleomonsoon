import pandas as pd
import numpy as np
import glob
import os


def remove_trailing_tabs(directory_path):
    # Get all txt files in the directory
    files = glob.glob(directory_path + "*.txt")

    for file in files:
        print(file)
        with open(file, 'r') as f:
            # Read lines and strip trailing whitespace/tabs
            cleaned_lines = [line.rstrip() for line in f.readlines()]

        # Write cleaned lines to a temporary file
        temp_file = file + "_temp"
        with open(temp_file, 'w') as f:
            f.write('\n'.join(cleaned_lines))

        # Replace the original file with the cleaned file
        os.remove(file)
        os.rename(temp_file, file)

    print("All files cleaned!")



def process_txt_data(directory_path):
    # Define metadata columns
    metadata_cols = ['longitude', 'latitude', 'proxy_archive', 'seasonality', 'reference', 'site_name', 'data_url', 'paleoData_TSid']
    df_metadata = pd.DataFrame(columns=metadata_cols + ['resolution'])
    df_values = pd.DataFrame(index=np.arange(0,2011))

    # Iterate over all txt files in the directory
    files = glob.glob(directory_path + "*.txt")

    for file in files:
        with open(file, 'r') as f:
            # Extract metadata from the first line
            metadata_line = f.readline().strip().split('\t')
            metadata = dict(zip(metadata_cols, metadata_line))
            dataset_name = file.split("/")[-1][:-4]
            
            # Extract data URL from the second line
            metadata['data_url'] = f.readline().strip()
            metadata['paleoData_TSid'] = dataset_name

            # Read the rest of the file into a dataframe
            data = pd.read_csv(f, sep='\t', header=None, names=['year', 'value'])
            #print(data)

            if data['value'].dtype == 'O' and data['value'].str.contains(',').any():
                data['value'] = data['value'].str.replace(',', '.').astype(float)

            # Calculate resolution
            resolution = np.abs(np.mean(np.diff(data['year'])))
            metadata['resolution'] = resolution

            # Append metadata to df_metadata
            df_metadata = df_metadata.append(metadata, ignore_index=True)

            # Interpolate values to match the basis_years
            rebased_values = np.interp(np.arange(0,2011), data['year'], data['value'], left=np.nan, right=np.nan)
            df_values[dataset_name] = rebased_values

    column_mapping = {
        'longitude': 'geo_meanLon',
        'latitude': 'geo_meanLat',
        'proxy_archive': 'archiveType',
        'site_name': 'geo_siteName',
        'data_url': 'pub1_dataUrl',
        'reference': 'reference'  # This remains the same but is included for clarity
    }
    df_metadata = df_metadata.rename(columns=column_mapping)

    # Save dataframes
    df_metadata.to_csv(directory_path + 'metadata.csv', index=False)
    df_values.to_csv(directory_path + 'values.csv', index=True)

    print(df_metadata)
    print(df_values)


#remove_trailing_tabs('/home/users/rz908899/mitre/paleo/hrnh2k/hydro_proxies/')
#remove_trailing_tabs('/home/users/rz908899/mitre/paleo/hrnh2k/temp_proxies/')

process_txt_data('/home/users/rz908899/mitre/paleo/hrnh2k/hydro_proxies/')
process_txt_data('/home/users/rz908899/mitre/paleo/hrnh2k/temp_proxies/')
