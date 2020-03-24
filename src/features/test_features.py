from src.features.build_features import (
    handle_zeros,
    kullback_leibler,
    resampled_teams
)
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
    assert all(np.isclose(grouped, 1))


def test_kullback_leibler_no_divergence():
    forecasts = np.array(
        [[0.1, 0.1],
         [0.9, 0.9]]
    )
    average = np.array([[0.1, 0.9]])
    assert (np.isclose(kullback_leibler(forecasts, average), 0)).all()


def test_kullback_leibler_non_negative():
    processed = preprocess(cycle3)
    processed = handle_zeros(processed)
    for question in processed.Question.unique():
        kiwi, kiwi_KL, mango, mango_KL = resampled_teams(processed,
                                                         'D',
                                                         question)
        print(question)
        assert all(kiwi_KL >= 0)


