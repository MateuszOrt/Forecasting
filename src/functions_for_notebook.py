import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX

import seaborn as sns
# from pylab import rcParams
from matplotlib import rc

# from tqdm.notebook import tqdm

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
# from collections import defaultdict

import math

# import warnings

# warnings.filterwarnings("ignore")  # avoid printing out absolute paths
import copy
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger


from lightning.pytorch.tuner import Tuner
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
# from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameter


def create_benchmark_data(data_points):
  # Parameters for the time series
  np.random.seed(42)  # For reproducibility
  n_points = data_points  # Number of data points
  t = np.arange(n_points)  # Time index

  # Generate trend (linear)
  trend = 0.05 * t

  # Generate seasonality (sinusoidal with fixed period)
  seasonal_period = 50
  seasonality = 2 * np.sin(2 * np.pi * t / seasonal_period)

  # Generate noise
  noise = np.random.normal(0, 0.5, n_points)

  # Combine all components
  time_series = trend + seasonality + noise

  # Create a DataFrame
  data = pd.DataFrame({'Time': t, 'Value': time_series})
  return data, time_series, t

# Train test split
# math.ceil round a number upward to its nearest integer
def data_transformation(data, size):
  training_data_len = math.ceil(len(data) * size)

  # Splitting the dataset # may use .iloc[:, :2] in diffrent case
  train_data = data[:training_data_len]
  test_data = data[training_data_len:]

  # Selecting values
  dataset_train = train_data.Value

  # Reshaping 1D to 2D array for MinMaxScaler
  #first arg num of rows, second number of elements in row -1 acts as One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
  dataset_train = np.reshape(dataset_train, (-1, 1))

  # Selecting Open Price values
  dataset_test = test_data.Value
  # Reshaping 1D to 2D array for MinMaxScaler
  #first arg num of rows, second number of elements in row -1 acts as One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.
  dataset_test = np.reshape(dataset_test, (-1, 1))

  scaler = MinMaxScaler(feature_range=(0, 1))
  # Scaling dataset
  scaled_train = scaler.fit_transform(dataset_train)

  # Normalizing values between 0 and 1
  scaled_test = scaler.fit_transform(dataset_test)
  return train_data, test_data, scaled_train, scaled_test, dataset_train, dataset_test, scaler


def create_sequences(data, sequence_length):
  X_train, y_train = [], []
  for i in range(len(data) - sequence_length):
    X_train.append(data[i:i + sequence_length])
    y_train.append(data[i + sequence_length])# Predicting the value right after the sequence
  X_train, y_train = np.array(X_train), np.array(y_train)
  return X_train, y_train


def print_gpu_memory():
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Cached memory: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
 
  
def forecast_model(model, device, X_test ,num_forecast_steps):
  # squeeze Remove axes of length one from a.
  sequence_to_plot = X_test.squeeze().cpu().numpy()
  historical_data = sequence_to_plot[-1]
  forecasted_values = []
  with torch.no_grad():
      for _ in range(num_forecast_steps):
          historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
          predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]
          forecasted_values.append(predicted_value)
          # Roll array elements along a given axis.
          # Elements that roll beyond the last position are re-introduced at the first.
          # Last value is being replaced by first one and then is being replaced wit newly predicted value
          historical_data = np.roll(historical_data, shift=-1)
          historical_data[-1] = predicted_value
  return forecasted_values

def model_eval(model, X_test, y_test, device):
  # Evaluate the model and calculate RMSE and R² score
  model.eval()
  with torch.no_grad():
      test_predictions = []
      for batch_X_test in X_test:
          batch_X_test = batch_X_test.to(device).unsqueeze(0)  # Add batch dimension
          test_predictions.append(model(batch_X_test).cpu().numpy().flatten()[0])

  test_predictions = np.array(test_predictions)

  # Calculate RMSE and R² score
  rmse = np.sqrt(mean_squared_error(y_test.cpu().numpy(), test_predictions))
  r2 = r2_score(y_test.cpu().numpy(), test_predictions)
  return rmse, r2

def forecast_model_on_raw_data(model, data_value, device, num_forecast_steps, prediction_start):
  # squeeze Remove axes of length one from a.
  historical_data = data_value[prediction_start:].squeeze().cpu().numpy()
  forecasted_values = []
  with torch.no_grad():
      for _ in range(num_forecast_steps):
          historical_data_tensor = torch.as_tensor(historical_data).view(1, -1, 1).float().to(device)
          predicted_value = model(historical_data_tensor).cpu().numpy()[0, 0]
          forecasted_values.append(predicted_value)
          # Roll array elements along a given axis.
          # Elements that roll beyond the last position are re-introduced at the first.
          # Last value is being replaced by first one and then is being replaced wit newly predicted value
          historical_data = np.roll(historical_data, shift=-1)
          historical_data[-1] = predicted_value
  return forecasted_values