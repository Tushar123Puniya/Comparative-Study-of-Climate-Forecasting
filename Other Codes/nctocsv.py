import warnings

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import xarray as xr
import pandas as pd

# Load the NetCDF datasets
ds = xr.open_dataset('/home/tushar/Weather Prediction/Data/Delhi Data/2016P1.nc')

latitudes = ds.variables['latitude'][:32].values  # Convert to numpy array (floats)
longitudes = ds.variables['longitude'][:64].values  # Convert to numpy array (floats)

print(latitudes)
print(longitudes)

global_dataframes = {}
for lat in latitudes:
    for lon in longitudes:
        global_dataframes[(lat, lon)] = pd.DataFrame()
        
def update_global_dataframe(lat, lon, df):
    global_dataframes[(lat, lon)] = pd.concat([global_dataframes[(lat, lon)], df], axis=0)
    
import xarray as xr

for year in range(2016,2024):
    # Load the NetCDF datasets
    ds1 = xr.open_dataset(f'/home/tushar/Weather Prediction/Data/Delhi Data/{year}P1.nc')
    ds2 = xr.open_dataset(f'/home/tushar/Weather Prediction/Data/Delhi Data/{year}P2.nc')
    ds3 = xr.open_dataset(f'/home/tushar/Weather Prediction/Data/Delhi Data/{year}P3.nc')
    ds4 = xr.open_dataset(f'/home/tushar/Weather Prediction/Data/Delhi Data/{year}S1.nc')
    ds5 = xr.open_dataset(f'/home/tushar/Weather Prediction/Data/Delhi Data/{year}S2.nc')
    
    import pandas as pd

    # Define the list of pressure levels
    pressure_levels = [50, 250, 500, 600, 700, 850, 925]

    # List of variables (features) to include with and without levels
    features_1 = [
        'u', 'r',
    ]
    features_2 = [
        'z', 'q',
    ]
    features_3 = [
        'v', 't',
    ]
    features_without_levels_1 = [
        'ssrd'
    ]
    features_without_levels_2 = [
        'u10', 'v10',
        't2m', 'lsm',
        'z'
    ]

    # Initialize an empty DataFrame for each location
    lats = latitudes
    lons = longitudes

    dataframes = {}
    for lat in latitudes:
        for lon in longitudes:
            dataframes[(lat, lon)] = pd.DataFrame()
            
    # Iterate over each coordinate and fill the DataFrames
    for lat in lats:
        for lon in lons:
            ds = ds1

            # Handle features with pressure levels
            for level in pressure_levels:
                    try:
                        for feature in features_1:
                            # Extract data for the given pressure level
                            data = ds.sel(latitude=lat, longitude=lon, pressure_level=level, method='nearest')

                            # Convert to pandas DataFrame
                            df = data[[feature]].to_dataframe().reset_index()

                            dataframes[(lat,lon)][f'{feature}_{level}'] = df[feature]

                    except KeyError as e:
                        print("Error in 1st nc file")
                        print(f"KeyError: {e} for location lat: {lat}, lon: {lon}")

            # Remove any duplicated rows if necessary
            dataframes[(lat, lon)].drop_duplicates(inplace=True)
            
    for lat in lats:
        for lon in lons:
            ds = ds2

            # Handle features with pressure levels
            for level in pressure_levels:
                    try:
                        for feature in features_2:
                            # Extract data for the given pressure level
                            data = ds.sel(latitude=lat, longitude=lon, pressure_level=level, method='nearest')

                            # Convert to pandas DataFrame
                            df = data[[feature]].to_dataframe().reset_index()

                            dataframes[(lat,lon)][f'{feature}_{level}'] = df[feature]

                    except KeyError as e:
                        print("Error in 2nd nc file")
                        print(f"KeyError: {e} for location lat: {lat}, lon: {lon}")

            # Remove any duplicated rows if necessary
            dataframes[(lat, lon)].drop_duplicates(inplace=True)
    
    for lat in lats:
        for lon in lons:
            ds = ds3

            # Handle features with pressure levels
            for level in pressure_levels:
                    try:
                        for feature in features_3:
                            # Extract data for the given pressure level
                            data = ds.sel(latitude=lat, longitude=lon, pressure_level=level, method='nearest')

                            # Convert to pandas DataFrame
                            df = data[[feature]].to_dataframe().reset_index()

                            dataframes[(lat,lon)][f'{feature}_{level}'] = df[feature]

                    except KeyError as e:
                        print("Error in 3rd nc file")
                        print(f"KeyError: {e} for location lat: {lat}, lon: {lon}")
        
            # Remove any duplicated rows if necessary
            dataframes[(lat, lon)].drop_duplicates(inplace=True)

    for lat in lats:
        for lon in lons:
            ds = ds4
            
            # Handle features without pressure levels
            for feature in features_without_levels_1:
                # Extract data for the given pressure level
                data = ds.sel(latitude=lat, longitude=lon, method='nearest')

                # Convert to pandas DataFrame
                df = data[[feature]].to_dataframe().reset_index()

                dataframes[(lat,lon)][f'{feature}'] = df[feature]

            # Remove any duplicated rows if necessary
            dataframes[(lat, lon)].drop_duplicates(inplace=True)

    for lat in lats:
        for lon in lons:
            ds = ds5

            # Handle features without pressure levels
            for feature in features_without_levels_2:
                # Extract data for the given pressure level
                data = ds.sel(latitude=lat, longitude=lon, method='nearest')

                # Convert to pandas DataFrame
                df = data[[feature]].to_dataframe().reset_index()

                dataframes[(lat,lon)][f'{feature}'] = df[feature]

            # Remove any duplicated rows if necessary
            dataframes[(lat, lon)].drop_duplicates(inplace=True)

    #         print(f"Updated DataFrame for location lat: {lat}, lon: {lon}")

    for (lat, lon), temp_df in dataframes.items():
            update_global_dataframe(lat, lon, temp_df)
            

        
for lat in latitudes:
    for lon in longitudes:
        global_dataframes[(lat,lon)]['Latitude'] = lat
        global_dataframes[(lat,lon)]['Longitude'] = lon
        
import os

# Define the folder path where CSV files will be saved
folder_path = 'Full_data_csv'

# Create the folder if it does not exist
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Save DataFrames to gzip-compressed CSV files in the specified folder
for (lat, lon), df in global_dataframes.items():  # Use 'global_dataframes' to save the updated DataFrames
    file_path = os.path.join(folder_path, f'{lat}_{lon}.csv.gz')
    df.to_csv(file_path, index=False, compression='gzip')

print(f"DataFrames saved to folder: {folder_path}")
