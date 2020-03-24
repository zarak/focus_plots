# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
from pathlib import Path
from dotenv import find_dotenv, load_dotenv


FILE_NAME = 'uniform_date.csv'


def take_last_forecast(df):
    """Get the last forecast if multiple forecasts have the same timestamp"""
    index_cols = ['Uniform Date Format', 'Question', 'Ordered.Bin.Number']
    agg_dict = {col: 'last' for col in df.columns if col not in index_cols}
    return df.groupby(index_cols, as_index=False).agg(agg_dict)


def preprocess(cycle3):
    cycle3 = cycle3.drop(columns={'Unnamed: 0', 'Timestamp'})
    cycle3 = cycle3.rename(columns={'Team.Name': 'TeamName'})
    cycle3 = cycle3.drop_duplicates()
    cycle3.loc[:,
               'Uniform Date Format'
               ] = pd.to_datetime(cycle3['Uniform Date Format'])
    cycle3 = cycle3.query('FairSkill != "######"')
    cycle3.loc[:, 'FairSkill'] = cycle3.FairSkill.astype(float)
    cycle3 = take_last_forecast(cycle3)
    cycle3 = cycle3.set_index("Uniform Date Format")
    cycle3 = cycle3.sort_index()
    return cycle3


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')
    cycle3 = pd.read_csv(Path(input_filepath) / FILE_NAME)
    cycle3 = preprocess(cycle3)
    cycle3.to_csv(Path(output_filepath) / 'cycle3.csv')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
