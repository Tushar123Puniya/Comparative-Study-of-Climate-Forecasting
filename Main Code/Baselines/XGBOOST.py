print("Importing libraries and setting seed")
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.data.dataloader")
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import random
import torch.optim as optim
import matplotlib.pyplot as plt
import psutil
import gc
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import json
from statsmodels.tsa.arima.model import ARIMA
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings("ignore")

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)             # For CPU
torch.cuda.manual_seed(seed)        # For a single GPU
torch.cuda.manual_seed_all(seed)    # For all GPUs (if you have multiple)
np.random.seed(seed)                # For NumPy
random.seed(seed)                   # For Python random module

# Ensure reproducibility in some CUDA operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

print("Setting device...")
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

print("Data Loading...")
import os
import pandas as pd

# parallel loading code.....
import dask.dataframe as dd

# Define folder and load files
from concurrent.futures import ProcessPoolExecutor

# Define the folder path
folder_path = "/home/tushar/Weather Prediction/Code/Baselines/Full_world_eight_year_Data"

# Get a sorted list of all CSV files in the folder
csv_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv.gz')])

# Define a function to read a CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Use ProcessPoolExecutor for parallel loading
dataframes = []
with ProcessPoolExecutor() as executor:
    dataframes = list(executor.map(load_csv, csv_files))

# Now 'dataframes' contains all the DataFrames from the CSV files in the folder

# Add the DateTime column to each DataFrame in the list--

num_rows = dataframes[0].shape[0]
start_date = "2016-01-01"
date_range = pd.date_range(start=start_date, periods=num_rows, freq="6H")

for df in dataframes:
    df['date_time'] = date_range

print("Calculating latitude_grid...")
# Extract all latitude values and find unique values
latitude_values = np.concatenate([df['Latitude'].values for df in dataframes])
unique_latitudes = np.unique(latitude_values)

# Sort the unique latitudes
sorted_latitudes = np.sort(unique_latitudes)

# Convert to a PyTorch tensor and move to the desired device in one step
latitude_tensor = torch.tensor(sorted_latitudes, device=device)

import pandas as pd

# Step 1: Define the date ranges for training, validation, and test sets
train_start = pd.Timestamp("2016-01-01")
train_end = pd.Timestamp("2020-12-31")
val_start = pd.Timestamp("2021-01-01")
val_end = pd.Timestamp("2021-12-31")
test_start = pd.Timestamp("2022-01-01")
test_end = pd.Timestamp("2023-12-31")

# Step 2: Split each DataFrame in the list into train, validation, and test sets
train_list = []
val_list = []
test_list = []

for df in dataframes:
    # Add "time_of_day" and "day_of_week" columns
    df['time_of_day'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek

    # Training set
    train_df = df[(df['date_time'] >= train_start) & (df['date_time'] <= train_end)]

    # Validation set
    val_df = df[(df['date_time'] >= val_start) & (df['date_time'] <= val_end)]

    # Test set
    test_df = df[(df['date_time'] >= test_start) & (df['date_time'] <= test_end)]

    # Append the split DataFrames to their respective lists
    train_list.append(train_df.reset_index(drop=True))
    val_list.append(val_df.reset_index(drop=True))
    test_list.append(test_df.reset_index(drop=True))

# Step 3: Drop unnecessary columns
def drop_columns(df_list, columns_to_drop):
    for i, df in enumerate(df_list):
        df_list[i] = df.drop(columns=columns_to_drop)

# Specify the columns to drop
columns_to_drop = ['date_time']

# Drop columns from each list
drop_columns(train_list, columns_to_drop)
drop_columns(val_list, columns_to_drop)
drop_columns(test_list, columns_to_drop)

# Get the current memory usage
memory_info = psutil.virtual_memory()

# Print the current used memory in GB
used_memory_gb = memory_info.used / (1024 ** 3)  # Convert bytes to GB

print(f"Used Memory: {used_memory_gb:.2f} GB")

del dataframes

gc.collect()

# Get the current memory usage
memory_info = psutil.virtual_memory()

# Print the current used memory in GB
used_memory_gb = memory_info.used / (1024 ** 3)  # Convert bytes to GB

print(f"Used Memory: {used_memory_gb:.2f} GB")

print("Defining Dataset Class...")

import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

class WeatherGridDataset(Dataset):
    def __init__(self, df_list, in_step, out_step, variable, num_workers=32, gap=1):
        self.in_step = in_step
        self.out_step = out_step
        self.variable = variable
        self.num_workers = num_workers
        self.gap = gap

        self.grid_data, self.lats, self.longs, self.variable_index, self.lsm_index, self.z_index = self._create_grid_data(df_list)
        self.inputs, self.outputs = self._prepare_sequences_parallel()
        
        self.input_mean, self.input_std = self._compute_statistics(self.inputs)
        self.inputs = self._normalize(self.inputs, self.input_mean, self.input_std)
        
        self.output_mean, self.output_std = self._compute_statistics(self.outputs)
        self.outputs = self._normalize(self.outputs, self.output_mean, self.output_std)

    def _create_grid_data(self, df_list):
        unique_lats = sorted(set(df['Latitude'].iloc[0] for df in df_list))
        unique_longs = sorted(set(df['Longitude'].iloc[0] for df in df_list))

        variable_index = df_list[0].columns.get_loc(self.variable)
        lsm_index = df_list[0].columns.get_loc('lsm')
        z_index = df_list[0].columns.get_loc('z')

        grid_data = np.empty((len(unique_lats), len(unique_longs)), dtype=object)

        for df in df_list:
            lat = df['Latitude'].iloc[0]
            long = df['Longitude'].iloc[0]

            lat_idx = unique_lats.index(lat)
            long_idx = unique_longs.index(long)

            grid_data[lat_idx, long_idx] = {
                'time_series': df.drop(columns=['Latitude', 'Longitude', 'lsm', 'z', 'time_of_day', 'day_of_week']).values,
                'latitude': lat,
                'lsm': df['lsm'].iloc[0],
                'z': df['z'].iloc[0]
            }

        return grid_data, unique_lats, unique_longs, variable_index, lsm_index, z_index

    def _prepare_sequences_parallel(self):
        num_lats = self.grid_data.shape[0]
        num_longs = self.grid_data.shape[1]
        num_time_steps = self.grid_data[0, 0]['time_series'].shape[0]

        inputs = []
        outputs = []

        def process_time_step(t):
            # Create tensors for all grid cells (lat, long) and all features (time-dependent + time-independent)
            input_tensor = np.zeros((num_lats, num_longs, self.in_step * (self.grid_data[0, 0]['time_series'].shape[1] + 3)))  # Including time-independent features
            output_tensor = np.zeros((self.out_step, num_lats, num_longs))

            for i in range(num_lats):
                for j in range(num_longs):
                    grid_cell = self.grid_data[i, j]

                    # Prepare input (flattened time sequence including time-dependent and time-independent features)
                    input_sequence = grid_cell['time_series'][t:t + self.in_step]  # Get the time-dependent part for the input sequence
                    latitude = np.full((1,), grid_cell['latitude'])  # Time-independent part: latitude
                    lsm = np.full((1,), grid_cell['lsm'])  # Time-independent part: lsm
                    z = np.full((1,), grid_cell['z'])  # Time-independent part: z

                    # Concatenate time-dependent and time-independent features for each timestamp
                    for time_idx in range(self.in_step):
                        # Flatten the time-dependent features and concatenate with time-independent features
                        input_tensor[i, j, time_idx * (input_sequence.shape[1] + 3):(time_idx + 1) * (input_sequence.shape[1] + 3)] = np.concatenate(
                            [input_sequence[time_idx], latitude, lsm, z]
                        )

                    # Predict at intervals of gap from the last input timestamp
                    for step in range(self.out_step):
                        out_idx = t + self.in_step - 1 + self.gap * (step + 1)
                        output_tensor[step, i, j] = grid_cell['time_series'][out_idx, self.variable_index]

            return input_tensor, output_tensor

        max_start = num_time_steps - (self.in_step - 1 + self.gap * self.out_step)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(process_time_step, range(max_start)))

        for inp, out in results:
            inputs.append(inp)
            outputs.append(out)

        return inputs, outputs

    def _compute_statistics(self, data):
        data_stack = np.stack(data, axis=0)
        mean = np.mean(data_stack, axis=(0, 1, 2))
        std = np.std(data_stack, axis=(0, 1, 2))
        return mean, std

    def _normalize(self, data, mean, std):
        std[std == 0] = 1
        return [(d - mean) / std for d in data]

    def inverse_transform(self, data, is_output=True):
        if is_output:
            return data * self.output_std + self.output_mean
        else:
            return data * self.input_std + self.input_mean

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_seq = torch.tensor(self.inputs[idx], dtype=torch.float32)
        output_seq = torch.tensor(self.outputs[idx], dtype=torch.float32)
        return input_seq, output_seq

import numpy as np
import xgboost as xgb

class LocationXGBoost:
    def __init__(self, out_steps=1, gap=0):
        self.out_steps = out_steps
        self.gap = gap
        self.models = [None for _ in range(out_steps)]

    def fit(self, X, y):
        """
        X: (B, in_steps) numpy array
        y: (B, out_steps) numpy array
        """
        for step in range(self.out_steps):
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=10)
            model.fit(X, y[:, step])
            self.models[step] = model

    def predict(self, X):
        """
        X: (B, in_steps) numpy array
        return: (B, out_steps) numpy array
        """
        preds = []
        for step in range(self.out_steps):
            pred_step = self.models[step].predict(X)
            preds.append(pred_step)
        preds = np.stack(preds, axis=1)
        return preds

print("Defining Custom Loss Function...")
# Latitude-Weighted RMSE Loss
class LatitudeWeightedRMSELoss(nn.Module):
    def __init__(self, latitude_grid):
        super(LatitudeWeightedRMSELoss, self).__init__()
        self.weights = torch.cos(torch.tensor(latitude_grid * np.pi / 180))
        self.weights = self.weights / self.weights .mean()  # L(i) = cos(H_i) / mean(cos(H_i))
    
    def forward(self, predictions, targets):
        squared_error = F.mse_loss(predictions, targets, reduction='none')
        weighted_squared_error = squared_error * self.weights.unsqueeze(0)  # For batch dimension
        weighted_squared_error = torch.mean(weighted_squared_error, dim=tuple(range(1, weighted_squared_error.ndim)))
        mean_weighted_squared_error = torch.sqrt(weighted_squared_error)
        return torch.mean(mean_weighted_squared_error)

# Sort the unique latitudes
sorted_latitudes = np.sort(unique_latitudes)

# Create a 2D array with each unique latitude repeated 64 times in each row
latitude_grid = np.tile(sorted_latitudes.reshape(-1, 1), (1, 64))

# Convert the grid to a PyTorch tensor
latitude_tensor = torch.tensor(latitude_grid)

# Assuming latitude_tesnor is to be used in a PyTorch model
latitude_tensor = latitude_tensor.to(device)

def latitude_weighted_rmse(predictions, targets, latitude_grid):
    """
    Computes the longitude-latitude weighted RMSE for multi-step forecasting.

    Args:
        predictions (torch.Tensor): Shape (k * data_size, lat, lon)
        targets (torch.Tensor): Shape (k * data_size, lat, lon)
        latitude_grid (np.array): Shape (lat,)
        k (int): Number of forecasted time steps.

    Returns:
        float: Weighted RMSE score.
    """
    latitude_grid = torch.tensor(latitude_grid)
    # Ensure latitude weights are on the same device as predictions
    lat_tensor = torch.tensor(latitude_grid * np.pi / 180, device=predictions.device)  # Convert to radians
    cos_lat = torch.cos(lat_tensor)  # cos(H_i)
    latitude_weights = cos_lat / cos_lat.mean()  # L(i) = cos(H_i) / mean(cos(H_i))
    
    # Compute squared error
    squared_error = (predictions - targets) ** 2
    latitude_grid = latitude_grid.unsqueeze(0).expand(3, -1, -1)
    
    # Apply latitude weighting
    weighted_squared_error = squared_error * latitude_weights.unsqueeze(0)  # Shape: (data_size, k, lat, lon)
    
    # Sum over k time steps
    summed_weighted_error = torch.mean(weighted_squared_error,dim=tuple(range(1, weighted_squared_error.ndim)))  # Shape: (data_size, lat, lon)

    sqrt_summed_weighted_error = torch.sqrt(summed_weighted_error)

    # Compute RMSE
    return torch.mean(sqrt_summed_weighted_error).item()

# ACC Metric
def calculate_acc(predictions, targets, mask=None):
    """
    Computes the longitude-latitude weighted RMSE for multi-step forecasting.

    Args:
        predictions (torch.Tensor): Shape (k * data_size, lat, lon)
        targets (torch.Tensor): Shape (k * data_size, lat, lon)
        latitude_grid (np.array): Shape (lat,)
        k (int): Number of forecasted time steps.

    Returns:
        float: Weighted RMSE score.
    """
    # Climatology per sample (mean over time steps)
    climatology = torch.mean(targets, dim=0, keepdim=True)  # (data_size, 1, lat, lon)

    # Calculate anomalies relative to climatology
    predictions_anomaly = predictions - climatology
    targets_anomaly = targets - climatology
    
    # Apply spatial mask, if provided
    if mask is not None:
        predictions_anomaly *= mask
        targets_anomaly *= mask

    # Compute the numerator: covariance between predictions and targets
    numerator = torch.sum(predictions_anomaly * targets_anomaly)

    # Compute the denominator: sqrt of variances for predictions and targets
    pred_variance = torch.sum(predictions_anomaly ** 2)
    target_variance = torch.sum(targets_anomaly ** 2)
    denominator = torch.sqrt(pred_variance) * torch.sqrt(target_variance)

    # Calculate ACC
    acc = numerator / denominator
    print(torch.all(predictions_anomaly == 0),torch.all(targets_anomaly == 0))
    print(numerator, denominator)
    return acc.item()

def evaluate_xgboost(location_models, loader, train_dataset, latitude_grid):
    all_preds = []
    all_targets = []

    for x, y in tqdm(loader, desc="Evaluating"):
        x = x.to(device).float()
        y = y.to(device).float()  # (B, lat, lon, out_steps)

        batch_preds = torch.zeros_like(y).to(device)

        for i in range(lat_size):
            for j in range(lon_size):
                model = location_models[(i, j)]
                num_features = x.size(-1) // in_steps
                x_loc_series = x[:, i, j, :].view(-1, in_steps, num_features)  # (B, in_steps, num_features)
                X = x_loc_series[:, :, -1].cpu().numpy()

                preds = model.predict(X)
                preds_tensor = torch.tensor(preds, device=device)
                
                # print(batch_preds.shape,preds_tensor.shape)
                batch_preds[:, :, i, j] = preds_tensor

        preds_inv = train_dataset.inverse_transform(batch_preds.cpu())
        targets_inv = train_dataset.inverse_transform(y.cpu())
        preds_inv = preds_inv.permute(0, 2, 3, 1)
        targets_inv = targets_inv.permute(0, 2, 3, 1)

        all_preds.append(preds_inv)
        all_targets.append(targets_inv)

    all_preds = torch.cat(all_preds, dim=0).permute(0, 3, 1, 2)  # (N, out_steps, lat, lon)
    all_targets = torch.cat(all_targets, dim=0).permute(0, 3, 1, 2)

    metrics_per_step = {
        'rmse': [], 'mape': [], 'acc': [], 'weighted_rmse': [],
    }

    for step in range(out_steps):
        preds_step = all_preds[:, step]
        targets_step = all_targets[:, step]

        rmse = torch.sqrt(torch.mean((preds_step - targets_step) ** 2)).item()
        mape = torch.mean(torch.abs((preds_step - targets_step) / (targets_step + 1e-6))).item() * 100
        acc = calculate_acc(preds_step, targets_step)
        weighted_rmse = latitude_weighted_rmse(preds_step, targets_step, latitude_grid)

        metrics_per_step['rmse'].append(rmse)
        metrics_per_step['mape'].append(mape)
        metrics_per_step['acc'].append(acc)
        metrics_per_step['weighted_rmse'].append(weighted_rmse)

    return metrics_per_step


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_var", type=str, required=True)
parser.add_argument("--out_steps", type=int, required=True)
parser.add_argument("--gap", type=int, required=True)
args = parser.parse_args()

pred_var = args.pred_var
in_steps = 3
out_steps = args.out_steps
gap = args.gap
# pred_var = "z_500"
# in_steps = 3
# out_steps = 1
# gap = 7

print("Creating the dataset")
# Create datasets and dataloaders
train_dataset = WeatherGridDataset(train_list,in_steps,out_steps,f'{pred_var}',gap=gap)
print("Freeing up memory from train_list...")
del train_list
val_dataset = WeatherGridDataset(val_list,in_steps,out_steps,f'{pred_var}',gap=gap)
print("Freeing up memory from val_list...")
del val_list
test_dataset = WeatherGridDataset(test_list,in_steps,out_steps,f'{pred_var}',gap=gap)
print("Freeing up memory from test_listmo...")
del test_list

gc.collect()

print("Creating the dataloaders")
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
print("Defining Model, Optimizer, and Criterion...")

# --------- Configuration ---------
lat_size = 32
lon_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"  # Use GPU if available

# --------- Initialize Models ---------
print("Initializing models...")

location_models = {
    (i, j): LocationXGBoost(out_steps=out_steps, gap=gap)
    for i in range(lat_size)
    for j in range(lon_size)
}

import torch
from tqdm import tqdm

# --------- Define Train Step ---------
def train_step_xgboost(i, j, x_batch, y_batch, num_features):
    model = location_models[(i, j)]

    # Extract the time series for location (i,j)
    x_loc_series = x_batch[:, i, j, :].view(-1, in_steps, num_features)  # (B, in_steps, num_features)

    # Use only the last feature
    X = x_loc_series[:, :, -1].cpu().numpy()  # (B, in_steps)

    # Handle the gap
    if model.gap > 0:
        y_shifted = torch.roll(y_batch[:, :, i, j], shifts=-model.gap, dims=0)
        y_shifted[-model.gap:] = 0  # Padding
    else:
        y_shifted = y_batch[:, :, i, j]

    y = y_shifted.cpu().numpy()  # (B, out_steps)

    # Fit the model
    model.fit(X, y)

# --------- Training Loop ---------
print("Training XGBoost models location by location...")

for x_batch, y_batch in tqdm(train_loader, desc="Training"):
    x_batch = x_batch.to(device)
    y_batch = y_batch.to(device)
    num_features = x_batch.size(-1) // in_steps

    for i in range(lat_size):
        for j in range(lon_size):
            train_step_xgboost(i, j, x_batch, y_batch, num_features)

print("Testing Starts...")

# Test Evaluation
_ = evaluate_xgboost(
    location_models=location_models,
    loader=test_loader,
    train_dataset=train_dataset,
    latitude_grid=latitude_grid,
)

import os
import json

def save_metrics_to_json(metrics_per_step, out_steps, pred_var, gap, filename="/home/tushar/Weather Prediction/Code/Main Code/Results/Direct Forecasting Results/XGBoost_Direct_Forecasting_result.json"):
    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # Prepare new entry to append
    new_entry = {
        "out_steps": out_steps,
        "gap": gap,
        "pred_var": pred_var,
        "metrics_per_step": []
    }

    for step in range(out_steps):
        step_metrics = {
            "lead_time": (step + 1) * gap,  # Reflects actual lead time
            "rmse": round(metrics_per_step["rmse"][step], 6),
            "mape": round(metrics_per_step["mape"][step], 6),
            "acc": round(metrics_per_step["acc"][step], 6),
            "weighted_rmse": round(metrics_per_step["weighted_rmse"][step], 6)
        }
        new_entry["metrics_per_step"].append(step_metrics)

    # Append the new entry
    existing_data.append(new_entry)

    # Save updated data back to the file
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Metrics appended to {os.path.abspath(filename)}")
    
save_metrics_to_json(_, out_steps=out_steps, pred_var=pred_var,gap=gap)

"""# ------------------------------------ End of the Notebook ----------------------------------"""
