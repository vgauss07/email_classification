import pandas as pd

from loguru import logger

from sklearn.preprocessing import LabelEncoder

from src.model.pipeline.collection import load_data, FILEPATH


def prepare_data():
    logger.info('Starting Preprocessing Pipeline')

    data = load_data(FILEPATH)

    clean_data = drop_missing_values(data)

    clean_data_1 = drop_duplicates(clean_data)

    final_df = encode_category_column(clean_data_1)

    return final_df


def drop_missing_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info('Dropping Missing Values')
    dataframe.dropna(inplace=True)
    return dataframe


def drop_duplicates(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info('Dropping Duplicates')
    dataframe.drop_duplicates(inplace=True)
    return dataframe


def encode_category_column(dataframe: pd.DataFrame) -> pd.DataFrame:
    logger.info('Encoding Category column')
    le = LabelEncoder()
    mappings = {}
    dataframe['Category'] = le.fit_transform(dataframe['Category'])
    mappings['Category'] = {label: code for label,
                            code in zip(le.classes_, le.transform(le.classes_))
                            }
    return mappings, dataframe


df = prepare_data()
print(df)
