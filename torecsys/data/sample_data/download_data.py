"""

"""

import logging
import math
import os
import warnings
import zipfile
from pathlib import Path

import requests

from torecsys.data.sample_data import __ml_size__, __jester_label__

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm

downloadable_data = ['ml-20m', 'ml-latest-small', 'ml-latest', 'ml-100k', 'ml-1m', 'ml-10m', 'bx']

logger = logging.getLogger(__name__)


def request_download(url: str, download_loc: str, block_size: int = 1024, is_zipped: bool = True, unzip_loc: str = ''):
    _wrote = 0

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get('content-length', 0))
    with open(download_loc, 'wb') as f:
        for data in tqdm(r.iter_content(block_size),
                         total=math.ceil(total_size / block_size),
                         unit='KB',
                         unit_scale=True):
            _wrote += len(data)
            f.write(data)

    if total_size != 0 and _wrote != total_size:
        raise RuntimeError(f'something went wrong during download from {url}.')

    if is_zipped:
        with zipfile.ZipFile(download_loc, 'r') as zip_file:
            zip_file.extractall(unzip_loc)
        os.remove(download_loc)


def download_data(data: str, **kwargs):
    """
    Download data

    Args:
        data:
        **kwargs:
    """
    directory = kwargs.get('directory', None)

    data = data.split('-')
    downloadable_data_name = data[0]
    downloadable_data_tags = '-'.join(data[1:]) if len(data) > 1 else ''

    if downloadable_data_name == 'ml':
        download_ml_data(size=downloadable_data_tags, directory=directory)
    elif downloadable_data_name == 'bx':
        download_bx_data(directory=directory)
    elif downloadable_data_name == 'jester':
        download_jester_data(label=downloadable_data_tags, directory=directory)
    else:
        raise ValueError('downloadable_data is not found. '
                         'Please check the list of torecsys.data.sample_data.download_data.downloadable_data')


def download_ml_data(size: str, directory: str = None) -> bool:
    """
    Download movielens data from grouplens, source: https://grouplens.org/datasets/movielens/.
    
    Args:
        size (str): Movielens dataset size, allows: 20m, latest-small, latest, 100k, 1m, 10m
        directory (str, optional): Directory to save downloaded data.
            Defaults to None.

    Raises:
        ValueError: when size is not in allowed values
        RuntimeError: when download face trouble
    """
    if size not in __ml_size__:
        raise ValueError(f'size must be in [{", ".join(__ml_size__)}].')

    if directory is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, 'sample_data')
    else:
        samples_dir = directory
    Path(samples_dir).mkdir(parents=True, exist_ok=True)

    unzip_folder = f'ml-{size}'
    zip_filename = f'ml-{size}.zip'
    zip_url = f'https://files.grouplens.org/datasets/movielens/{zip_filename}'
    zip_file_loc = os.path.join(samples_dir, zip_filename)

    request_download(zip_url, zip_file_loc, unzip_loc=samples_dir)
    logger.info(f'Finished: file {unzip_folder} is downloaded to the directory: '
                f'{os.path.join(samples_dir, unzip_folder)}')

    return True


def download_criteo_data(directory: str = None):
    """
    Download criteo data from Criteo AI Lab,
    source: https://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/.
    
    Args:
        directory (str, optional): Directory to save downloaded data. Defaults to None.
    """
    # set directory name and create directory if not exist
    if directory is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, 'sample_data')
    else:
        samples_dir = directory

    Path(samples_dir).mkdir(parents=True, exist_ok=True)

    unzip_folder = 'dac'
    zip_filename = 'dac.tar.gz'
    zip_url = 'https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz'
    zip_file_loc = os.path.join(samples_dir, zip_filename)

    request_download(zip_url, zip_file_loc, unzip_loc=samples_dir)
    logger.info(f'Finished: file {unzip_folder} is downloaded to the directory: '
                f'{os.path.join(samples_dir, unzip_folder)}')


def download_bx_data(directory: str = None):
    """
    Download Book-Crossing data to directory ../sample_data/,
    source: https://www2.informatik.uni-freiburg.de/~cziegler/BX/
    
    Args:
        directory (str, optional): Directory to save downloaded data.
            Defaults to None.

    Raises:
        ValueError: when size is not in allowed values
        RuntimeError: when download face trouble
    """
    if directory is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, 'sample_data')
    else:
        samples_dir = directory

    unzip_dir = os.path.join(samples_dir, 'bx')
    Path(unzip_dir).mkdir(parents=True, exist_ok=True)
    zip_filename = 'bx.zip'
    zip_url = 'https://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip'
    zip_file_loc = os.path.join(samples_dir, zip_filename)

    request_download(zip_url, zip_file_loc, unzip_loc=samples_dir)
    logger.info(f'Finished: file {zip_filename} is downloaded to the directory: '
                f'{os.path.join(samples_dir, zip_filename)}')


def download_jester_data(label: str, directory: str = None):
    """
    Download data from jester

    Args:
        label (str): Label of jester dataset.
        directory (str): Directory of download target.
    """
    if label not in __jester_label__:
        raise ValueError(f'label must be in [{", ".join(__jester_label__)}].')

    if directory is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, 'sample_data')
    else:
        samples_dir = directory

    # make directory of unzip_dir
    unzip_dir = os.path.join(samples_dir, f'jester-{label}')
    Path(unzip_dir).mkdir(parents=True, exist_ok=True)

    # set file name to download data
    zip_filename = f'jester-{label}.zip'
    zip_url = f'https://goldberg.berkeley.edu/jester-data/jester-data-{label}.zip'
    zip_file_loc = os.path.join(samples_dir, zip_filename)

    request_download(zip_url, zip_file_loc, unzip_loc=samples_dir)
    logger.info(f'Finished: file {zip_filename} is downloaded to the directory: {os.path.join(zip_file_loc)}')
