# Libraries Installing
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
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from tqdm import tqdm

pred_var = 't2m'

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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("Data Loading...")
import os
import pandas as pd

# Define folder and load files
from concurrent.futures import ProcessPoolExecutor

# Define the folder path
folder_path = "/home/anonymus/Weather Prediction/Data/Delhi_Data"

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

print("Freeing up memory of dataframes...")
del dataframes

print("Defining Dataset Class...")
# Dataset Class

class CustomDataset(Dataset):
    def __init__(self, in_steps, df_list, out_steps, fixed_features,  # Changed fixed_feature to fixed_features
                 exclude_features=["Latitude", "Longitude", "day_of_week", "time_of_day"], epsilon=1e-8):
        """
        Args:
            in_steps (int): Number of input time steps.
            df_list (list of pd.DataFrame): List of dataframes, each corresponding to a different location.
            out_steps (int): Number of output time steps.
            fixed_features (list of str): List of column names for the output data (target features).
            exclude_features (list of str): List of features to exclude from normalization.
            epsilon (float): Small value to prevent division by zero during standardization.
        """
        self.in_steps = in_steps
        self.df_list = df_list
        self.out_steps = out_steps
        self.fixed_features = fixed_features  # Now accepts multiple fixed features
        self.exclude_features = exclude_features
        self.epsilon = epsilon

        # Check if all fixed_features exist in all dataframes
        for df in df_list:
            for feature in fixed_features:
                if feature not in df.columns:
                    raise ValueError(f"Fixed feature '{feature}' not found in dataframe columns.")

        # Collect the names of the features to normalize
        self.feature_columns = [col for col in df_list[0].columns if col not in exclude_features]

        # Calculate the mean and standard deviation for the features to be normalized across all dataframes
        all_data = np.concatenate([df[self.feature_columns].values for df in df_list], axis=0)
        self.feature_means = np.mean(all_data, axis=0)
        self.feature_stds = np.std(all_data, axis=0)

        # For output features (fixed_features), calculate mean and std for each target feature
        self.output_means = []
        self.output_stds = []
        for feature in fixed_features:
            output_data = np.concatenate([df[feature].values for df in df_list])
            self.output_means.append(np.mean(output_data))
            self.output_stds.append(np.std(output_data))

        # Add epsilon to avoid division by zero
        self.feature_stds = np.where(self.feature_stds == 0, self.epsilon, self.feature_stds)
        self.output_stds = np.where(np.array(self.output_stds) == 0, self.epsilon, self.output_stds)

    def __len__(self):
        # The length of the dataset is the number of samples we can generate
        return len(self.df_list[0]) - self.in_steps - self.out_steps + 1

    def __getitem__(self, idx):
        # Create tensors to hold the input and output data
        num_nodes = len(self.df_list)
        num_features = len(self.df_list[0].columns)
        num_output_features = len(self.fixed_features)  # Number of output features

        # Initialize the input and output tensors
        inputs = torch.zeros((self.in_steps, num_nodes, num_features))
        outputs = torch.zeros((self.out_steps, num_nodes, num_output_features))  # Adjust for multiple outputs

        for i, df in enumerate(self.df_list):
            # Extract the relevant slice of data for this dataframe
            start_idx = idx
            end_idx = start_idx + self.in_steps
            out_end_idx = end_idx + self.out_steps

            try:
                # Extract input data
                input_data = df.iloc[start_idx:end_idx].copy()

                # Standardize the features except for the excluded ones
                input_data[self.feature_columns] = (input_data[self.feature_columns] - self.feature_means) / self.feature_stds

                # Convert to tensor and assign to inputs
                inputs[:, i, :] = torch.tensor(input_data.values, dtype=torch.float32)

                # Extract and standardize the output data for each fixed feature
                for j, feature in enumerate(self.fixed_features):
                    output_data = df.iloc[end_idx:out_end_idx][feature].values
                    output_data = (output_data - self.output_means[j]) / self.output_stds[j]

                    # Reshape and assign to outputs tensor (now multiple output features)
                    outputs[:, i, j] = torch.tensor(output_data, dtype=torch.float32)

            except ValueError as e:
                print(e)
                continue

        return inputs, outputs

    def inverse_transform_outputs(self, outputs):
        """Inverse transform the normalized outputs for all fixed features."""
        outputs_np = outputs.numpy()
        for j, (mean, std) in enumerate(zip(self.output_means, self.output_stds)):
            outputs_np[:, :, j] = (outputs_np[:, :, j] * std) + mean
        return torch.tensor(outputs_np, dtype=torch.float32)

# in_step, out_step and the prediction feature defining
in_steps = 3
out_steps = 1
fixed_feature = [pred_var]

print("Creating Dataset...")
train_dataset = CustomDataset(in_steps, train_list, out_steps, fixed_feature)
print("Freeing memory of train_list...")
del train_list
val_dataset = CustomDataset(in_steps, val_list, out_steps, fixed_feature)
print("Freeing memory of val_list...")
del val_list
test_dataset = CustomDataset(in_steps, test_list, out_steps, fixed_feature)
print("Freeing memory of test_list...")
del test_list

print("Creating data loaders...")
from torch.utils.data import DataLoader

# Define batch size
batch_size = 16

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=32,pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=32,pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=32,pin_memory=True)

print("Defining custom loss...")
# Custom Loss Function
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

# Function to inverse-transform predictions and targets to the original scale
def inverse_transform(data, mean, std):
    return data * std + mean

print("Defining Metrices...")
# Metrices
# Longitude-Latitude Weighted RMSE Metric
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

print("Defining Model Architecture...")
# Model Architecture
# Staeformer Model for handling multiple features as output
class AttentionLayer(nn.Module):
    """Perform attention across the -2 dim (the -1 dim is `model_dim`).

    Make sure the tensor is permuted to correct shape before attention.

    E.g.
    - Input shape (batch_size, in_steps, num_nodes, model_dim).
    - Then the attention will be performed across the nodes.

    Also, it supports different src and tgt length.

    But must `src length == K length == V length`.

    """

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()

        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask

        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)

        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        # Q    (batch_size, ..., tgt_length, model_dim)
        # K, V (batch_size, ..., src_length, model_dim)
        batch_size = query.shape[0]
        tgt_length = query.shape[-2]
        src_length = key.shape[-2]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # Qhead, Khead, Vhead (num_heads * batch_size, ..., length, head_dim)
        query = torch.cat(torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(
            -1, -2
        )  # (num_heads * batch_size, ..., head_dim, src_length)

        attn_score = (
            query @ key
        ) / self.head_dim**0.5  # (num_heads * batch_size, ..., tgt_length, src_length)

        if self.mask:
            mask = torch.ones(
                tgt_length, src_length, dtype=torch.bool, device=query.device
            ).tril()  # lower triangular part of the matrix
            attn_score.masked_fill_(~mask, -torch.inf)  # fill in-place

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value  # (num_heads * batch_size, ..., tgt_length, head_dim)
        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads = model_dim)

        out = self.out_proj(out)

        return out


class SelfAttentionLayer(nn.Module):
    def __init__(
        self, model_dim, feed_forward_dim=2048, num_heads=8, dropout=0, mask=False
    ):
        super().__init__()

        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        # x: (batch_size, ..., length, model_dim)
        residual = x
        out = self.attn(x, x, x)  # (batch_size, ..., length, model_dim)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)  # (batch_size, ..., length, model_dim)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out

class SPAETransformer(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=3,
        out_steps=1,
        steps_per_day=4,
        input_dim=6,
        output_dim=1,  # Modified: This can now be a list of target features
        input_embedding_dim=48,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=24,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim  # Handle multiple output dims
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)

        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if self.spatial_embedding_dim > 0:
            self.latlon_embeddings = nn.Linear(2, self.spatial_embedding_dim)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(in_steps, num_nodes, adaptive_embedding_dim))
            )
            
        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim  # Adjusted for multiple targets
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, output_dim)  # Adjusted for multiple targets

        
        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]
        if self.tod_embedding_dim > 0:
            tod = x[..., -1]
        if self.dow_embedding_dim > 0:
            dow = x[..., -2]/6
        if self.spatial_embedding_dim > 0:
            lat = x[..., -4]
            lon = x[..., -3]

        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]

        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * (self.steps_per_day) / 24).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            latlon = torch.stack([lat, lon], dim=-1)  # Combine lat and lon
            latlon_emb = self.latlon_embeddings(latlon)  # Get the embedding
            features.append(latlon_emb)

        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        for attn in self.attn_layers_t:
            x = attn(x, dim=1)
        for attn in self.attn_layers_s:
            x = attn(x, dim=2)
        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim  # Adjusted for multiple targets
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else: # Hidden layers not added
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, len(output_dim)) Adjusted for multiple targets

        return out

# Assuming your dataset class stores output_mean and output_std for each feature
output_mean = train_dataset.output_means  # Shape: (3,)
output_std = train_dataset.output_stds  # Shape: (3,)

print('Model declaration...')
# Initialize the model
model = SPAETransformer(
    num_nodes=8*8,
    in_steps=in_steps,
    out_steps=out_steps,
    input_dim=49,
    output_dim=1  # Predicting single features
)

model = model.to(device)


# Using the RMSE loss criterion
criterion = LatitudeWeightedRMSELoss(latitude_tensor)

# Optimizer settings
optimizer = optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-5)

# Learning rate schedulers
warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
main_scheduler = CosineAnnealingLR(optimizer, T_max=45)

print('Hyperparameters setting...')
# Training Hyperparameters and Variables
num_epochs = 100
patience = 100
start_epoch = 0
best_val_loss = float('inf')
patience_counter = 0
accumulation_steps = 16 
train_losses, val_losses = [], []

print('Defining Evaluation function...')
def evaluate(model, loader, latitude_grid, output_mean, output_std):
    """
    Evaluate the model with Latitude-Weighted RMSE, RMSE, MAPE, and ACC metrics.

    Args:
    - model: PyTorch model to evaluate.
    - loader: DataLoader for evaluation data.
    - latitude_grid: Array of latitude values corresponding to the spatial grid.
    - output_mean: List of mean values for inverse transformation.
    - output_std: List of standard deviations for inverse transformation.

    Returns:
    - eval_loss: Average Latitude-Weighted RMSE loss over the dataset.
    - metrics: Dictionary containing Latitude-Weighted RMSE, RMSE, MAPE, and ACC for each feature.
    """
    model.eval()
    eval_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device).float()
            y = y.to(device).float()

            outputs = model(x)
            outputs = outputs.view(y.shape)

            # Latitude-Weighted RMSE Loss
            latitude_weights = torch.cos(torch.tensor(latitude_grid * np.pi / 180, device=outputs.device))
            squared_error = F.mse_loss(outputs, y, reduction='none')
            weighted_squared_error = squared_error * latitude_weights.unsqueeze(0)
            mean_weighted_squared_error = weighted_squared_error.mean()
            batch_loss = torch.sqrt(mean_weighted_squared_error)

            eval_loss += batch_loss.item()

            # Store predictions and targets
            all_preds.append(outputs)
            all_targets.append(y)

    # Concatenate tensors
    all_preds = torch.cat(all_preds).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()

    # Inverse transform predictions and targets to original scale
    num_output_features = all_preds.shape[-1]
    for i in range(num_output_features):
        all_preds[..., i] = inverse_transform(all_preds[..., i], output_mean[i], output_std[i])
        all_targets[..., i] = inverse_transform(all_targets[..., i], output_mean[i], output_std[i])

    # Compute metrics for each feature
    metrics = {}
    for i in range(num_output_features):
        preds_flat = all_preds[..., i].flatten()
        targets_flat = all_targets[..., i].flatten()

        # Longitude-Latitude Weighted RMSE
        lw_rmse = latitude_weighted_rmse(
            torch.tensor(preds_flat).to(device),
            torch.tensor(targets_flat).to(device),
            latitude_grid
        )

        # RMSE
        rmse = np.sqrt(np.mean((preds_flat - targets_flat) ** 2))

        # MAPE
        mape = np.mean(np.abs((targets_flat - preds_flat) / (targets_flat + 1e-8))) * 100

        # ACC
        predictions_tensor = torch.tensor(all_preds[..., i]).to(device)
        targets_tensor = torch.tensor(all_targets[..., i]).to(device)
        acc = calculate_acc(predictions_tensor, targets_tensor)

        # Store metrics
        metrics[f'feature_{i}'] = {
            'lw_rmse': lw_rmse,
            'rmse': rmse,
            'mape': mape,
            'acc': acc
        }

    return eval_loss / len(loader), metrics

print('Checkpoint checking...')
# Path to the checkpoint
checkpoint_path = f'/home/anonymus/Weather Prediction/Code/SPAE_Transformer/Delhi Checkpoints/{pred_var}_main_model_last_model.pth'

# Check if the checkpoint file exists
if os.path.exists(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path,map_location = device)

    # Restore the model state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Restore the optimizer state
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Restore the schedulers
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=5)
    warmup_scheduler.load_state_dict(checkpoint['scheduler_state_dict']['warmup'])

    main_scheduler = CosineAnnealingLR(optimizer, T_max=45)
    main_scheduler.load_state_dict(checkpoint['scheduler_state_dict']['main'])

    # Restore other training parameters
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    best_val_loss = checkpoint['loss']  # Best validation loss so far

    print(f"Resumed training from epoch {start_epoch} with best validation loss {best_val_loss:.4f}")

else:
    print(f"No checkpoint found at {checkpoint_path}. Starting training from scratch.")

print(f'Results for {pred_var}')
print('Training starts...')
# Training Loop
from tqdm import tqdm
end_epoch = start_epoch+num_epochs

for epoch in range(start_epoch, end_epoch + 1):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{end_epoch}", leave=False) as tbar:
        for batch_idx, (x, y) in tbar:
            x = x.to(device).float()
            y = y.to(device).float()

            # Forward pass
            outputs = model(x)
            outputs = outputs.view(y.shape)
            loss = criterion(outputs, y) / accumulation_steps

            # Backward pass
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            tbar.set_postfix({'Batch Loss': loss.item()})

    train_loss = epoch_loss / len(train_loader)
    train_losses.append(train_loss)

    # Validation phase
    val_loss, val_metrics = evaluate(model, val_loader, latitude_tensor, output_mean, output_std)
    val_losses.append(val_loss)

    for feature, metrics in val_metrics.items():
        print(f'Validation Metrics for {feature}:')
        print(f'  Latitude Weighted RMSE: {metrics["lw_rmse"]}, ACC: {metrics["acc"]}, RMSE: {metrics["rmse"]}, MAPE: {metrics["mape"]}')

    print(f'Epoch {epoch+1}/{end_epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    # Scheduler step
    if epoch < 5:
        warmup_scheduler.step()
    else:
        main_scheduler.step()

    # Save last epoch checkpoint
    checkpoint_last = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': {
            'warmup': warmup_scheduler.state_dict(),
            'main': main_scheduler.state_dict()
        },
        'loss': min(val_loss, best_val_loss),
        'latest_loss': train_loss
    }
    torch.save(checkpoint_last, f'Delhi Checkpoints/{pred_var}_main_model_last_model.pth')

    # Save best model checkpoint
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        checkpoint_best = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': {
                'warmup': warmup_scheduler.state_dict(),
                'main': main_scheduler.state_dict()
            },
            'loss': val_loss
        }
        torch.save(checkpoint_best, f'Delhi Checkpoints/{pred_var}_main_model_best_model.pth')

    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

# Load the best model weights for evaluation
print('Evaluating the model on test data')
best_model_path = f"/home/anonymus/Weather Prediction/Code/Main Model/Delhi Checkpoints/{pred_var}_main_model_best_model.pth"

# Load the checkpoint
checkpoint = torch.load(best_model_path,map_location = device)

# Extract only the model's state_dict from the checkpoint
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()

# Evaluation phase with RMSE and inverse transformation for test data
test_loss, test_metrics = evaluate(model, test_loader, latitude_tensor, output_mean, output_std)

# Log metrics for each output feature
for feature, metrics in test_metrics.items():
    print(f'Test Metrics for {feature}:')
    print(f'  Latitude Weighted RMSE: {metrics["lw_rmse"]}, ACC: {metrics["acc"]}, RMSE: {metrics["rmse"]}, MAPE: {metrics["mape"]}')

print(f'Test Loss (LWRMSE): {test_loss}')

# Plotting the training and validation loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot as an image file
plt.savefig(f'loss_plot_main_model_{pred_var}.png')

