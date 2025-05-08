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


print("Defining Modle Architecture....")

# --------- LSTM Model ---------
print("Declaring LSTM Model...")
class LocationLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps):
        super(LocationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        # x: (batch_size, seq_len=3, input_size=49)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last time step
        return self.fc(out)
    
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
    return acc.item()

print("Defining Evaluation Function...")

def evaluate(location_models, loader, criterion, train_dataset, latitude_grid, device="cpu"):
    eval_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()  # shape: (B, lat, lon, in_channels)
            y = y.to(device).float()  # shape: (B, lat, lon, out_steps)

            batch_preds = torch.zeros_like(y)

            for i in range(lat_size):
                for j in range(lon_size):
                    model = location_models[(i, j)]
                    model.eval()

                    x_loc = x[:, i, j, :]  # shape: (B, in_channels)
                    y_loc = y[:, :, i, j]  # shape: (B, out_steps)

                    x_loc = x_loc.view(-1, in_steps, num_features)  # (B, in_steps, num_features)
                    preds_loc = model(x_loc)  # (B, out_steps)

                    batch_preds[:, :, i, j] = preds_loc

            loss = criterion(batch_preds, y)
            eval_loss += loss.item()

            # Inverse transform predictions and targets
            preds = train_dataset.inverse_transform(batch_preds.cpu())
            targets = train_dataset.inverse_transform(y.cpu())
            preds = preds.permute(0, 2, 3, 1)  # (B, lat, lon, out_steps)
            targets = targets.permute(0, 2, 3, 1)
            all_preds.append(preds)
            all_targets.append(targets)

    # Convert to (N, out_steps, lat, lon)
    all_preds = torch.cat(all_preds, dim=0).permute(0, 3, 1, 2)  # (N, lat, lon, out_steps) → (N, out_steps, lat, lon)
    all_targets = torch.cat(all_targets, dim=0).permute(0, 3, 1, 2)

    out_steps = all_preds.shape[1]

    metrics_per_step = {
        'rmse': [],
        'mape': [],
        'acc': [],
        'weighted_rmse': [],
    }
    
    for step in range(out_steps):
        preds_step = all_preds[:, step]       # (N, lat, lon)
        targets_step = all_targets[:, step]   # (N, lat, lon)

        rmse = torch.sqrt(torch.mean((preds_step - targets_step) ** 2)).item()
        mape = torch.mean(torch.abs((preds_step - targets_step) / (targets_step + 1e-6))).item() * 100
        acc = calculate_acc(preds_step, targets_step)
        weighted_rmse = latitude_weighted_rmse(preds_step, targets_step, latitude_grid)

        metrics_per_step['rmse'].append(rmse)
        metrics_per_step['mape'].append(mape)
        metrics_per_step['acc'].append(acc)
        metrics_per_step['weighted_rmse'].append(weighted_rmse)

    return eval_loss / len(loader), metrics_per_step


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

print("Defining the Model, optimizer and criterion")
# --------- Configuration ---------
lat_size = 32
lon_size = 64
num_features = 49
hidden_size = 32
epochs = 20
batch_size = 16  # per location
learning_rate = 0.001

# --------- Initialize Models and Optimizers ---------
print("Initializing models...")
location_models = {
    (i, j): LocationLSTM(num_features, hidden_size, out_steps).to(device)
    for i in range(lat_size)
    for j in range(lon_size)
}

optimizers = {
    (i, j): torch.optim.Adam(location_models[(i, j)].parameters(), lr=learning_rate)
    for i in range(lat_size)
    for j in range(lon_size)
}

criterion = nn.MSELoss()
from concurrent.futures import ThreadPoolExecutor

# --------- Training Loop (Parallel) ---------
print("Training models using train_loader (parallelized)...")

def train_step(i, j, x_loc, y_loc):
    model = location_models[(i, j)]
    optimizer = optimizers[(i, j)]

    x_loc = x_loc.view(-1, in_steps, num_features)  # (B, in_steps, num_features)

    model.train()
    optimizer.zero_grad()
    pred = model(x_loc)
    loss = criterion(pred, y_loc)
    loss.backward()
    optimizer.step()
    return loss.item()

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    epoch_loss = 0
    for x_batch, y_batch in tqdm(train_loader, desc="Batches"):
        # x_batch: (B, lat, lon, in_channels)
        # y_batch: (B, lat, lon, out_steps)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        tasks = []
        with ThreadPoolExecutor(max_workers=32) as executor:  # Adjust based on your CPU/GPU capacity
            for i in range(lat_size):
                for j in range(lon_size):
                    x_loc = x_batch[:, i, j, :]  # (B, in_channels)
                    y_loc = y_batch[:,:, i, j]  # (B, out_steps)
                    tasks.append(executor.submit(train_step, i, j, x_loc, y_loc))

            for future in tasks:
                epoch_loss += future.result()

    print(f"Epoch {epoch+1} Loss: {epoch_loss / (lat_size * lon_size):.4f}")

def save_metrics_to_json(metrics_per_step, out_steps, pred_var, gap, filename="/home/tushar/Weather Prediction/Code/Main Code/Results/Direct Forecasting Results/LSTM_Direct_Forecasting_result.json"):
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

print("Testing Starts...")
# Test Evaluation
test_loss, _ = evaluate(
    location_models=location_models,
    loader=test_loader,
    criterion=criterion,
    train_dataset=train_dataset,
    latitude_grid=latitude_grid,
    device=device
)

save_metrics_to_json(_, out_steps=out_steps, pred_var=pred_var,gap=gap)

"""# ------------------------------------ End of the Notebook ----------------------------------"""