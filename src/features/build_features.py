import pandas as pd
import numpy as np


def handle_zeros(cycle3):
    C = 0.005
    cycle3.loc[:, 'Forecast'] = cycle3.Forecast.replace(0, C)
    return cycle3


def kullback_leibler(forecasts, average):
    return -average @ (np.log2(forecasts) - np.log2(average.T))


def KL_apply(question):
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
