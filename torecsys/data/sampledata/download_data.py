from . import __ml_size__
import math
import os
from pathlib import Path
import requests
import warnings
import zipfile

# ignore import warnings of the below packages
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm

def download_ml_data(size : str,
                     dir  : str = None):
    r"""Download movielens data from grouplens.
    Source: https://grouplens.org/datasets/movielens/.
    
    Args:
        size (str): Movielens dataset size, allows: 20m, latest-small, latest, 100k, 1m, 10m
        dir (str, optional): Directory to save downloaded data. Default to None.

    Raises:
        ValueError: when size is not in allowed values 
        RuntimeError: when download face trouble 
    """
    # check if the size is allowed
    if size not in __ml_size__:
        raise ValueError("size must be in [%s]." % (", ".join(__ml_size__)))
    
    # set directory name and create directory if not exist
    if dir is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, "sample_data")
    else:
        samples_dir = dir
    Path(samples_dir).mkdir(parents=True, exist_ok=True)

    # set file name to download data
    unzip_folderdir = ("ml-%s" % size)
    zip_filename = ("ml-%s.zip" % size)
    zip_fileurl = ("http://files.grouplens.org/datasets/movielens/%s" % zip_filename)
    zip_fileloc = os.path.join(samples_dir, zip_filename)

    # streaming to iterate over the response
    r = requests.get(zip_fileurl, stream=True)
    
    # total size in bytes
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    wrote = 0
    
    # save the file
    with open(zip_fileloc, "wb") as f:
        for data in tqdm(r.iter_content(block_size), 
                         total=math.ceil(total_size / block_size),
                         unit="KB",
                         unit_scale=True):
            wrote += len(data)
            f.write(data)
    
    if total_size !=0 and wrote != total_size:
        raise RuntimeError("something went wrong.")
    
    # unzip and remove the file
    with zipfile.ZipFile(zip_fileloc, "r") as zip_file:
        zip_file.extractall(samples_dir)
    os.remove(zip_fileloc)

    print("Finished: file %s is downloaded to the directory: %s" % 
          (unzip_folderdir, os.path.join(samples_dir, unzip_folderdir)))

def download_criteo_data(dir: str = None):
    r"""Download criteo data from Critel AI Lab.
    Source: https://labs.criteo.com/2014/02/download-kaggle-display-advertising-challenge-dataset/.
    
    Args:
        dir (str, optional): Directory to save downloaded data. Defaults to None.
    """
    # set directory name and create directory if not exist
    if dir is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, "sample_data")
    else:
        samples_dir = dir
    Path(samples_dir).mkdir(parents=True, exist_ok=True)

    # set file name to download data
    unzip_folderdir = "dac"
    zip_filename = "dac.tar.gz"
    zip_fileurl = "https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz"
    zip_fileloc = os.path.join(samples_dir, zip_filename)

    # streaming to iterate over the response
    r = requests.get(zip_fileurl, stream=True)

    # total size in bytes
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024
    wrote = 0

    # save the file
    with open(zip_fileloc, "wb") as f:
        for data in tqdm(r.iter_content(block_size), 
                         total=math.ceil(total_size / block_size),
                         unit="KB",
                         unit_scale=True):
            wrote += len(data)
            f.write(data)
    
    if total_size !=0 and wrote != total_size:
        raise RuntimeError("something went wrong.")
    
    # unzip and remove the file
    with zipfile.ZipFile(zip_fileloc, "r") as zip_file:
        zip_file.extractall(samples_dir)
    os.remove(zip_fileloc)

    print("Finished: file %s is downloaded to the directory: %s" % 
          (unzip_folderdir, os.path.join(samples_dir, unzip_folderdir)))
    