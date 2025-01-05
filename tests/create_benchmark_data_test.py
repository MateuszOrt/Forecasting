from functions_for_notebook import create_benchmark_data


def test_create_benchmark_data():
    data, time_series, t = create_benchmark_data(data_points=100)
    assert len(data["Value"]) == 100
    assert len(time_series) == 100
    assert len(t) == 100
