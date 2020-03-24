import pandas as pd
import numpy as np


def handle_zeros(cycle3):
    C = 0.005
    cycle3.loc[:, 'Forecast'] = cycle3.Forecast.replace(0, C)
    return cycle3


def KL_divergence(df):
    return np.sum(
        df['Average.Forecast'] * np.log2(df['Average.Forecast'] /
                                         df['Forecast']))


def resample_KL(df):
    mu_b = df.groupby(
        ['Question', 'Ordered.Bin.Number']
    ).Forecast.mean().reset_index().rename(
        columns={'Forecast': 'Average.Forecast'}
    )
    f_t_b = df.reset_index().groupby(
        ['Uniform Date Format', 'Question', 'Ordered.Bin.Number']
    ).apply(lambda x: x['Forecast']).reset_index()
    if not f_t_b.empty:
        merged = pd.merge(
            mu_b, f_t_b, on=['Question', 'Ordered.Bin.Number'], how='right')
        return merged.groupby(
            ['Question', 'Ordered.Bin.Number']
        ).apply(KL_divergence).mean()


def resampled_teams(processed, resampling_period, question):
    kiwi = processed.query(
        "Question == @question & TeamName == 'Kiwi'"
    ).groupby(pd.Grouper(freq=resampling_period,
                         label='right')).apply(resample_KL)
    print(kiwi)
    kiwi_KL = processed.query(
        "Question == @question & TeamName == 'Kiwi'"
    ).groupby(pd.Grouper(
        freq=resampling_period, label='right')).apply(resample_KL).values

    mango = processed.query(
        "Question == @question & TeamName == 'Mango'"
    ).groupby(pd.Grouper(freq=resampling_period,
                         label='right')).apply(resample_KL)
    mango_KL = processed.query(
        "Question == @question & TeamName == 'Mango'"
    ).groupby(pd.Grouper(freq=resampling_period,
                         label='right')).apply(resample_KL).values

    return kiwi, kiwi_KL, mango, mango_KL
