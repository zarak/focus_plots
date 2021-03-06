import pandas as pd
import numpy as np


def handle_zeros(cycle3, C=0.005):
    """Laplace smoothing to handle 0 probabilities"""
    cycle3.loc[:, 'Forecast'] = cycle3.Forecast.replace(0, C)
    sums = cycle3.groupby(
        ['Uniform Date Format', 'Question']
    ).Forecast.sum().reset_index().rename(columns={'Forecast': 'Total'})
    merged = cycle3.merge(sums, on=['Question', 'Uniform Date Format'])
    merged.loc[:, 'Forecast'] = merged.Forecast / merged.Total
    return merged


def kullback_leibler(forecasts: np.ndarray, average: np.ndarray) -> np.ndarray:
    """
    Kullback-Leibler divergence
    Args:
        forecasts: An m by n matrix where m is the number
            of bins and n is the number of forecasters.
        average: A 1 by m row vector of floats.
    """
    return -average @ (np.log2(forecasts) - np.log2(average.T))


def KL_apply(question):
    """
    Compute average KL-divergence of all forecasters in a given time interval
    """
    if not question.empty:
        average_forecast = question.groupby(
            ['Ordered.Bin.Number']
        ).Forecast.mean().reset_index()
        forecasts = question.pivot_table(
            values='Forecast',
            index='Ordered.Bin.Number',
            columns='Forecaster.ID'
        ).values
        question_avg = average_forecast.Forecast.values.reshape(1, -1)
        divergences = kullback_leibler(forecasts, question_avg)
        return np.mean(divergences)


def resampled_teams(processed, resampling_period, question):
    """Subset by team, time interval, and question"""
    kiwi = processed.query(
        "Question == @question & TeamName == 'Kiwi'"
    ).reset_index().groupby(
        pd.Grouper(
            freq=resampling_period,
            label='right',
            key='Uniform Date Format',
        )).mean().reset_index()

    kiwi_KL = processed.query(
        "Question == @question & TeamName == 'Kiwi'"
    ).reset_index().groupby(
        pd.Grouper(
            freq=resampling_period,
            label='right',
            key='Uniform Date Format',
        )).apply(KL_apply)

    mango = processed.query(
        "Question == @question & TeamName == 'Mango'"
    ).reset_index().groupby(
        pd.Grouper(
            freq=resampling_period,
            label='right',
            key='Uniform Date Format',
        )).mean().reset_index()

    mango_KL = processed.query(
        "Question == @question & TeamName == 'Mango'"
    ).reset_index().groupby(
        pd.Grouper(
            freq=resampling_period,
            label='right',
            key='Uniform Date Format',
        )).apply(KL_apply)

    return kiwi, kiwi_KL, mango, mango_KL
