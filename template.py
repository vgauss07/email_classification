import logging
import os

from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

list_of_files = [
    "project/src/config/__init__.py",
    "project/src/__init__.py",
    "project/src/config/.env",
    "project/src/logs",
    "project/src/db/db_model.py",
    "project/src/model/models",
    "project/src/runner.py",
    "project/src/model/pipeline/collection.py",
    "project/src/model/pipeline/preparation.py",
    "project/src/model/pipeline/model.py",
    "project/Makefile",
    "project/setup.cfg",
    "project/research/test.ipynb",
    "project/research/experiment.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir} for the file: {filename}')

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, 'w') as f:
            pass
            logging.info(f'Creating empty file: {filepath}')

    else:
        logging.info(f'{filename} already exists')
