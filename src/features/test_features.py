from src.features.build_features import handle_zeros, kullback_leibler
from src.data.make_dataset import preprocess
from pathlib import Path
import pandas as pd
import numpy as np

FILE_NAME = 'uniform_date.csv'
PATH = Path('../data/raw')
cycle3 = pd.read_csv(PATH / FILE_NAME)


def test_handle_zeros():
    processed = preprocess(cycle3)
    processed = handle_zeros(processed)
    forecasts = processed.Forecast
    grouped = processed.groupby(
        ['Uniform Date Format', 'Question']
    ).Forecast.sum().values
    assert all(np.logical_and(0 <= forecasts, forecasts <= 1))
    assert all(grouped == 1)


def test_kullback_leibler():
    pass

