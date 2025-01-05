import pytest
from functions_for_notebook import data_transformation, create_benchmark_data
import pandas as pd
import numpy as np

def test_data_transformation():
    data, time_series, t=create_benchmark_data(data_points=100)
    train_data, test_data, scaled_train, scaled_test, dataset_train, dataset_test, scaler=data_transformation(data, .8)
    assert type(train_data) == pd.DataFrame
    assert type(test_data) == pd.DataFrame
    assert type(scaled_train) == np.ndarray
    assert type(scaled_test) == np.ndarray
    assert type(dataset_train) == np.ndarray
    assert type(dataset_test) == np.ndarray