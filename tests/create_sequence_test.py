from functions_for_notebook import create_sequences, create_benchmark_data


def test_create_sequences():
    data, time_series, t = create_benchmark_data(data_points=100)
    X_train, y_train = create_sequences(data["Value"], 30)
    assert X_train.shape == (70, 30)
    assert y_train.shape == (70,)
