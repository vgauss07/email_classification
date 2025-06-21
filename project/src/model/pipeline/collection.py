import pandas as pd

from loguru import logger

FILEPATH = '/Users/rev.dr.sylviablessings/email_classification/project/research/mail_data.csv'

def load_data(FILEPATH):
    logger.info(f'Loading csv file at path {FILEPATH}')
    return pd.read_csv(FILEPATH)
