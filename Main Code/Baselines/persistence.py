import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import torch

# ----------------- Metric Functions -----------------

def latitude_weighted_rmse(predictions, targets, latitude_grid):
    lat_tensor = torch.tensor(latitude_grid * np.pi / 180, device=predictions.device)
    cos_lat = torch.cos(lat_tensor)
    latitude_weights = cos_lat / cos_lat.mean()
    squared_error = (predictions - targets) ** 2
    weighted_squared_error = squared_error * latitude_weights.view(1, -1, 1)
    rmse = torch.sqrt(weighted_squared_error.mean(dim=(1, 2)))
    return rmse.mean().item()

def calculate_acc(predictions, targets, mask=None):
    climatology = torch.mean(targets, dim=0, keepdim=True)
    predictions_anomaly = predictions - climatology
    targets_anomaly = targets - climatology
    if mask is not None:
        predictions_anomaly *= mask
        targets_anomaly *= mask
    numerator = torch.sum(predictions_anomaly * targets_anomaly)
    denominator = torch.sqrt(torch.sum(predictions_anomaly ** 2) * torch.sum(targets_anomaly ** 2))
    acc = numerator / (denominator + 1e-6)
    return acc.item()

def mape(preds, targets):
    return (torch.abs((targets - preds) / (targets + 1e-6))).mean().item() * 100

# ----------------- Persistence Model Evaluation -----------------

def evaluate_persistence_model(data_dir, pred_var, gap, start_year, end_year, eval_years, timestamps_per_day,lat=32,lon=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Read all data and extract latitudes
    print("Reading data from CSVs...")
    filenames = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv.gz")], key=lambda x: int(x.split(".")[0]))
    num_locations = len(filenames)
    
    data = []
    latitudes = []
    
    for filename in tqdm(filenames):
        df = pd.read_csv(os.path.join(data_dir, filename))
        data.append(df[pred_var].values)
        latitudes.append(df['Latitude'].values[0])  # latitude is constant for each file

    data = np.stack(data, axis=1)  # shape: (time, location)
    latitude_grid = np.array(latitudes).reshape(lat, lon)
    latitude_mean = np.mean(latitude_grid, axis=1)

    # Step 2: Build timestamp list (assumes regular 6-hour interval)
    total_days = (datetime(end_year, 12, 31) - datetime(start_year, 1, 1)).days + 1
    dates = [datetime(start_year, 1, 1) + timedelta(hours=6 * i) for i in range(total_days * timestamps_per_day)]

    # Step 3: Get predictions and targets using Persistence Model
    print("Generating predictions using Persistence model...")
    preds, targets = [], []
    for t in range(1, len(dates) - gap):  # Starting from index 1 for persistence model (using previous timestamp)
        future_dt = dates[t + gap]
        if future_dt.year not in eval_years:
            continue
        
        # Persistence model: Use the previous value as the prediction
        preds.append(data[t - 1])  # Previous timestamp's value
        targets.append(data[t + gap])  # Actual value at the future timestamp

    preds = np.stack(preds, axis=0).reshape(-1, lat, lon)
    targets = np.stack(targets, axis=0).reshape(-1,lat, lon)

    preds_tensor = torch.tensor(preds, dtype=torch.float32, device=device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32, device=device)

    # Step 4: Compute metrics
    metrics = {
        "rmse": torch.sqrt(torch.mean((preds_tensor - targets_tensor) ** 2)).item(),
        "mape": mape(preds_tensor, targets_tensor),
        "acc": calculate_acc(preds_tensor, targets_tensor),
        "weighted_rmse": latitude_weighted_rmse(preds_tensor, targets_tensor, latitude_mean)
    }

    return metrics

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pred_var", type=str, required=True)
parser.add_argument("--out_steps", type=int, required=True)
parser.add_argument("--gap", type=int, required=True)
args = parser.parse_args()

pred_var = args.pred_var
out_steps = args.out_steps
gap = args.gap

# 9. Run evaluation
metrics = evaluate_persistence_model(
    data_dir="/home/tushar/Weather Prediction/Code/Baselines/Full_world_eight_year_Data",
    pred_var=pred_var,
    gap=gap,
    start_year=2016,
    end_year=2023,
    eval_years=[2022, 2023],
    timestamps_per_day=4,
    lat=32,
    lon=64
)

import json

def save_metrics_to_json(metrics_per_step, out_steps, pred_var, gap, filename="/home/tushar/Weather Prediction/Code/Main Code/Results/Direct Forecasting Results/Persistence_Direct_Forecasting_result.json"):
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

    new_entry["metrics_per_step"].append({
    "lead_time": gap,
    "rmse": round(metrics_per_step["rmse"], 6),
    "mape": round(metrics_per_step["mape"], 6),
    "acc": round(metrics_per_step["acc"], 6),
    "weighted_rmse": round(metrics_per_step["weighted_rmse"], 6)
    })

    # Append the new entry
    existing_data.append(new_entry)

    # Save updated data back to the file
    with open(filename, "w") as f:
        json.dump(existing_data, f, indent=4)

    print(f"Metrics appended to {os.path.abspath(filename)}")

save_metrics_to_json(metrics, out_steps=out_steps, pred_var=pred_var, gap=gap)
