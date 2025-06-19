import pandas as pd

from loguru import logger


def load_data(FILEPATH):
    logger.info(f'Loading csv file at path {FILEPATH}')
    return pd.read_csv(FILEPATH)
