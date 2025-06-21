import pandas as pd
import pickle as pk

from loguru import logger

from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV

from config.model import model_settings
from model.pipeline.preparation import prepare_data


def build_model():
    logger.info('Starting up Model Building Pipeline')

    # Load processed data
    df = prepare_data()

    # identify X and y
    X, y = _get_X_y(df)

    # split dataset
    X_train, X_test, y_train, y_test = _split_train_test(X, y)

    # vectorize data
    X_train_tfidf, X_test_tfidf = _vectorize_data(X_train, X_test)

    # train model
    rF = _train_random_forest(X_train_tfidf, y_train)

    # evaluate model
    _evaluate_model(rF, X_test_tfidf, y_test)

    # save model
    _save_model(rF)


def _get_X_y(dataframe: pd.DataFrame,
             col_X: str = 'Message',
             col_y: str = 'Category'):
    logger.info(f'Defining X and Y variables.'
                f'\nX vars: {col_X}\ny var: {col_y}')
    return dataframe[col_X], dataframe[col_y]


def _split_train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=32)
    return X_train, X_test, y_train, y_test


def _vectorize_data(X_train: pd.DataFrame,
                    X_test: pd.DataFrame):
    logger.info('Start vectorizing data with tfidf')

    tfidf = TfidfVectorizer()
    X_train_tfidf = tfidf.fit_transform(X_train).toarray()
    X_test_tfidf = tfidf.transform(X_test).toarray()

    logger.info('Successfully vectorized data with tfidf')

    return X_train_tfidf, X_test_tfidf


def _train_random_forest(X_train_tfidf, y_train):
    logger.info('Start training the random forest model')

    rf = RandomForestClassifier(random_state=32)

    params_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': randint(10, 50),
        'min_samples_split': randint(2, 10),
        'min_samples_leaf': randint(1, 5),
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        estimator=rf,
        param_distributions=params_dist,
        n_iter=5,
        cv=5,
        verbose=2,
        random_state=32,
        scoring='accuracy'
    )

    logger.debug(f'params_dist = {params_dist}')

    model_grid = random_search.fit(X_train_tfidf, y_train)

    logger.info('Done training models')

    return model_grid.best_estimator_


def _evaluate_model(model, X_test_tfidf, y_test):
    logger.info(f'Evaluting model performance. Score = '
                f'{model.score(X_test_tfidf, y_test)}')
    return model.score(X_test_tfidf, y_test)


def _save_model(model):
    logger.info(f'Saving the model: '
                f'{model_settings.model_path}/'
                f'{model_settings.model_name}')

    pk.dump(model,
            open(f'{model_settings.model_path}/'
                 f'{model_settings.model_name}', 'wb'))


# test
build_model()
