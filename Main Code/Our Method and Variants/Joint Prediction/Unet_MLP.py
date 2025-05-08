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
folder_path = "/home/tushar/Weather Prediction/Data/Delhi_Data"
lat = 8    # while running on full world data set it to 32
lon = 8    # while running on full world data set it to 64

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
    def __init__(self, df_list, in_step, out_step, pred_vars, num_workers=32, gap=1):
        self.in_step = in_step
        self.out_step = out_step
        self.pred_vars = pred_vars  # List of variables to predict
        self.num_workers = num_workers
        self.gap = gap

        self.grid_data, self.lats, self.longs, self.variable_indices, self.lsm_index, self.z_index = self._create_grid_data(df_list)
        self.inputs, self.outputs = self._prepare_sequences_parallel()

        self.input_mean, self.input_std = self._compute_statistics(self.inputs)
        self.inputs = self._normalize(self.inputs, self.input_mean, self.input_std)

        self.output_mean, self.output_std = self._compute_statistics(self.outputs)
        self.outputs = self._normalize(self.outputs, self.output_mean, self.output_std)

    def _create_grid_data(self, df_list):
        unique_lats = sorted(set(df['Latitude'].iloc[0] for df in df_list))
        unique_longs = sorted(set(df['Longitude'].iloc[0] for df in df_list))

        # Get indices for each variable in pred_vars
        variable_indices = [df_list[0].columns.get_loc(var) for var in self.pred_vars]
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

        return grid_data, unique_lats, unique_longs, variable_indices, lsm_index, z_index

    def _prepare_sequences_parallel(self):
        num_lats = self.grid_data.shape[0]
        num_longs = self.grid_data.shape[1]
        num_time_steps = self.grid_data[0, 0]['time_series'].shape[0]

        inputs = []
        outputs = []

        def process_time_step(t):
            input_tensor = np.zeros((num_lats, num_longs, self.in_step * self.grid_data[0, 0]['time_series'].shape[1] + 3))
            output_tensor = np.zeros((self.out_step, len(self.pred_vars), num_lats, num_longs))  # Multiple target variables

            for i in range(num_lats):
                for j in range(num_longs):
                    grid_cell = self.grid_data[i, j]

                    # Prepare input (flattened time sequence)
                    input_sequence = grid_cell['time_series'][t:t + self.in_step].flatten()
                    latitude = np.full((1,), grid_cell['latitude'])
                    lsm = np.full((1,), grid_cell['lsm'])
                    z = np.full((1,), grid_cell['z'])

                    input_tensor[i, j] = np.concatenate([input_sequence, latitude, lsm, z])

                    # Predict at intervals of gap from the last input timestamp
                    for step in range(self.out_step):
                        out_idx = t + self.in_step - 1 + self.gap * (step + 1)
                        # Collect all the predicted variables for the output tensor
                        for var_idx, variable_index in enumerate(self.variable_indices):
                            output_tensor[step, var_idx, i, j] = grid_cell['time_series'][out_idx, variable_index]

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


import torch
import torch.nn as nn
import torch.nn.functional as F

class PeriodicPadding(nn.Module):
    def __init__(self, pad_width, pad_height):
        super().__init__()
        self.pad_width = pad_width
        self.pad_height = pad_height

    def forward(self, x):
        return F.pad(x, (self.pad_width, self.pad_width, self.pad_height, self.pad_height), mode='circular')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, hidden_dim, dropout, pad_width=1, pad_height=1):
        super().__init__()
        self.padding = PeriodicPadding(pad_width, pad_height)
        self.conv1 = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_dim)
        self.relu = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(hidden_dim)
            ) if in_channels != hidden_dim else nn.Identity()
        )

    def forward(self, x):
        residual = x
        out = self.padding(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.padding(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(residual)
        out = self.relu(out)
        out = self.dropout(out)
        return out


class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads=2, dropout=0.1):
        super().__init__()
        # self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)

        # Xavier initialization
        nn.init.xavier_uniform_(self.attn.in_proj_weight)
        if self.attn.in_proj_bias is not None:
            nn.init.zeros_(self.attn.in_proj_bias)
        nn.init.xavier_uniform_(self.attn.out_proj.weight)
        if self.attn.out_proj.bias is not None:
            nn.init.zeros_(self.attn.out_proj.bias)

    def forward(self, x):
        b, lat, lon, f = x.shape
        # x = self.norm(x)
        x = x.view(b, lat * lon, f)
        x, _ = self.attn(x, x, x)
        x = x.view(b, lat, lon, f)
        return x


class MLP(nn.Module):
    def __init__(self, in_dim, hidden1=128, hidden2=64, dropout=0.1, out_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, out_dim)
        )

    def forward(self, x):
        b, lat, lon, f = x.shape
        x = x.reshape(-1, f)  # (B * lat * lon, f)
        x = self.fc(x)
        x = x.view(b, lat, lon, -1)
        return x


class UnetMLP(nn.Module):
    def __init__(self, in_channels=141, hidden_dim=128, channel_mults=[1, 2, 2],
                 num_blocks=2, dropout=0.1, num_heads=3, final_channels=1,lat=32,lon=64):
        super().__init__()
        self.in_channels = in_channels
        self.padding = PeriodicPadding(pad_width=3, pad_height=3)
        self.lat = lat
        self.lon = lon
        self.initial_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=0, bias=False)
        self.down_blocks = nn.ModuleList()
        current_channels = hidden_dim

        for mult in channel_mults:
            out_channels = hidden_dim * mult
            self.down_blocks.append(self._make_layer(current_channels, out_channels, num_blocks, dropout))
            current_channels = out_channels

        self.bottleneck = self._make_layer(current_channels, current_channels, num_blocks, dropout)

        self.up_blocks = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_channels = hidden_dim * mult
            self.up_blocks.append(self._make_layer(current_channels + out_channels, out_channels, num_blocks, dropout))
            current_channels = out_channels

        self.final_conv = nn.Conv2d(current_channels, in_channels, kernel_size=1, stride=1, padding=0)

        self.attention = SelfAttention(dim=in_channels, num_heads=num_heads, dropout=dropout)
        self.mlp = MLP(in_dim=in_channels, hidden1=128, hidden2=64, dropout=dropout, out_dim=final_channels)

    def _make_layer(self, in_channels, out_channels, num_blocks, dropout):
        layers = [ResidualBlock(in_channels, out_channels, dropout)]
        for _ in range(num_blocks - 1):
            layers.append(ResidualBlock(out_channels, out_channels, dropout))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.padding(x)
        x = self.initial_conv(x)

        skips = []
        for down in self.down_blocks:
            x = down(x)
            skips.append(x)
            x = F.avg_pool2d(x, kernel_size=2, stride=2)

        x = self.bottleneck(x)

        for up, skip in zip(self.up_blocks, reversed(skips)):
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if skip.shape[2:] != x.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")
            x = torch.cat([x, skip], dim=1)
            x = up(x)

        x = self.final_conv(x)
        x = F.interpolate(x, size=(self.lat, self.lon), mode="bilinear", align_corners=False)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)

        # x = self.attention(x)
        x = self.mlp(x)
        x = x.squeeze(-1)  # (B, H, W)
        return x


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
latitude_grid = np.tile(sorted_latitudes.reshape(-1, 1), (1, lon))

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
# Evaluation Function
def evaluate(model, loader, criterion, latitude_grid):
    model.eval()
    eval_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()
            outputs = model(x).view(y.shape)

            # Latitude-weighted RMSE loss
            loss = criterion(outputs, y)
            eval_loss += loss.item()

            # Inverse-transform predictions and targets
            preds = train_dataset.inverse_transform(outputs.cpu())
            targets = train_dataset.inverse_transform(y.cpu())

            all_preds.append(preds)
            all_targets.append(targets)

    # Convert to tensors with shape (N, out_steps, lat, lon)
    all_preds = torch.tensor(np.concatenate(all_preds, axis=0))
    all_targets = torch.tensor(np.concatenate(all_targets, axis=0))

    out_steps = all_preds.shape[2]

    metrics_per_step = {
        'rmse': [],
        'mape': [],
        'acc': [],
        'weighted_rmse': [],
    }

    for step in range(out_steps):
        preds_step = all_preds[:,:, step]       # shape: (N, lat, lon)
        targets_step = all_targets[:,:, step]   # shape: (N, lat, lon)

        # RMSE
        rmse = torch.sqrt(torch.mean((preds_step - targets_step) ** 2)).item()
        metrics_per_step['rmse'].append(rmse)

        # MAPE
        mape = torch.mean(torch.abs((preds_step - targets_step) / (targets_step + 1e-6))).item() * 100
        metrics_per_step['mape'].append(mape)

        # ACC
        acc = calculate_acc(preds_step, targets_step)
        metrics_per_step['acc'].append(acc)

        # Weighted RMSE
        weighted_rmse = latitude_weighted_rmse(preds_step, targets_step, latitude_grid)
        metrics_per_step['weighted_rmse'].append(weighted_rmse)

    return eval_loss / len(loader), metrics_per_step


print("Defining Training Function...")
# Training Loop
def train_model(model, train_loader, val_loader, optimizer, criterion, latitude_grid):
    num_epochs = 200
    patience = 10
    accumulation_steps=8
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    # Load last model if it exists
    flag = False
    if flag and os.path.exists(last_model_path):
        print("Loading last saved checkpoint...")
        checkpoint = torch.load(last_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_val_loss = checkpoint['best_val_loss']
        patience_counter = checkpoint['patience_counter']
        start_epoch = checkpoint['epoch'] + 1
        print('Resuming training from epoch', start_epoch)
    else:
        print('Starting training from scratch')
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        
        for i, (x, y) in enumerate(train_loader_tqdm):
            x = x.to(device).float()
            y = y.to(device).float()

            outputs = model(x)
            print(y.shape)
            print(outputs.shape)
            outputs = outputs.view(y.shape)
            loss = criterion(outputs, y)
            loss = loss / accumulation_steps  # Normalize loss for accumulation
            loss.backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * accumulation_steps  # Scale back loss
            train_loader_tqdm.set_postfix(loss=epoch_loss / (i + 1))

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation with tqdm
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        val_loss, _ = evaluate(model, val_loader, criterion, latitude_grid)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save last model checkpoint
        last_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        torch.save(last_checkpoint, last_model_path)

        # Save best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_counter': patience_counter,
                'train_losses': train_losses,
                'val_losses': val_losses
            }
            torch.save(best_checkpoint, best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    return train_losses, val_losses

# import argparse

# parser = argparse.ArgumentParser()
# parser.add_argument("--out_steps", type=int, required=True)
# parser.add_argument("--gap", type=int, required=True)
# args = parser.parse_args()

# pred_var = ['z_500','t2m','t_850']
# in_steps = 3
# out_steps = args.out_steps
# gap = args.gap

pred_var = ['z_500','t2m','t_850']
in_steps = 3
out_steps = 3
gap = 1

print("Creating the dataset")
# Create datasets and dataloaders
train_dataset = WeatherGridDataset(train_list,in_steps,1,pred_var,gap=gap)
print("Freeing up memory from train_list...")
del train_list
val_dataset = WeatherGridDataset(val_list,in_steps,1,pred_var,gap=gap)
print("Freeing up memory from val_list...")
del val_list
test_dataset = WeatherGridDataset(test_list,in_steps,1,pred_var,gap=gap)
print("Freeing up memory from test_listmo...")
del test_list

gc.collect()

print("Creating the dataloaders")
from torch.utils.data import DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print("Defining the Model, optimizer and criterion")
# Model, Optimizer, Criterion
model = UnetMLP(final_channels=out_steps,lat=lat,lon=lon)
model = model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-5)
criterion = LatitudeWeightedRMSELoss(latitude_tensor)

best_model_path = f'/home/tushar/Weather Prediction/Code/Main Code/Checkpoints/unet_mlp_best_model{pred_var}.pth'
last_model_path = f'/home/tushar/Weather Prediction/Code/Main Code/Checkpoints/unet_mlp_last_model{pred_var}.pth'
    
  
print("Training Starts...")
# Train Model
train_losses, val_losses = train_model(model, train_loader, val_loader, optimizer, criterion, latitude_grid)

# Load the best model
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['model_state_dict'])

import os
import json

def save_metrics_to_json(metrics_per_step, out_steps, pred_var, gap, filename="/home/tushar/Weather Prediction/Code/Main Code/Results/Joint Forecasting Results/Unet_MLP_alltogether_result.json"):
    # Load existing data if file exists
    if os.path.exists(filename):
        with open(filename, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    # Prepare new entry to append
    new_entry = {
        "gap": gap,
        "pred_var": pred_var,
        "metrics_per_step": []
    }

    dicti = {
        1:"z_500",
        2:"t2m",
        3:"t_850"
    }
    
    for step in range(out_steps):
        step_metrics = {
            "variable": dicti[(step+1)],  # Reflects actual lead time
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
test_loss, _ = evaluate(model, test_loader, criterion, latitude_grid)

save_metrics_to_json(_, out_steps=out_steps, pred_var=pred_var,gap=gap)

"""# ------------------------------------ End of the Notebook ----------------------------------"""