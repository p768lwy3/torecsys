from . import __ml_size__, __jester_label__
import math
import os
from pathlib import Path
import requests
import warnings
import zipfile

# Ignore import warnings of below packages
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from tqdm.autonotebook import tqdm

downloadable_data = [
    "ml-20m", "ml-latest-small", "ml-latest", "ml-100k", "ml-1m", "ml-10m", "bx"
]

def download_data(downloadable_data: str, 
                  **kwargs):
    
    downloadable_data = downloadable_data.split("-")
    downloadable_data_name = downloadable_data[0]
    if len(downloadable_data) > 1:
        downloadable_data_tags = "-".join(downloadable_data[1:])
    
    if downloadable_data_name == "ml":
        download_ml_data(size=downloadable_data_tags, dir=kwargs.get("dir", None))
    elif downloadable_data_name == "bx":
        download_bx_data(dir=kwargs.get("dir", None))
    elif downloadable_data_name == "jester":
        download_jester_data(label=downloadable_data_tags, dir=kwargs.get("dir", None))
    else:
        raise ValueError(
            "downloadable_data is not found. Please check the list of" + 
            "torecsys.data.sampledata.dowload_data.downloadable_data")

def download_ml_data(size : str,
                     dir  : str = None):
    r"""Download movielens data from grouplens.
    Source: https://grouplens.org/datasets/movielens/.
    
    Args:
        size (str): Movielens dataset size, allows: 20m, latest-small, latest, 100k, 1m, 10m
        dir (str, optional): Directory to save downloaded data. 
            Defaults to None.

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

def download_bx_data(dir : str = None):
    r"""Download Book-Crossing data to directory ../sample_data/,
    and the source url is : http://www2.informatik.uni-freiburg.de/~cziegler/BX/
    
    Args:
        dir (str, optional): Directory to save downloaded data. 
            Defaults to None.

    Raises:
        ValueError: when size is not in allowed values
        RuntimeError: when download face trouble
    """
    # set directory name and create directory if not exist
    if dir is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, "sample_data")
    else:
        samples_dir = dir
    
    # make directory of unzip_dir
    unzip_dir = os.path.join(samples_dir, "bx")
    Path(unzip_dir).mkdir(parents=True, exist_ok=True)

    # set file name to download data
    zip_filename = "bx.zip"
    zip_fileurl = "http://www2.informatik.uni-freiburg.de/~cziegler/BX/BX-CSV-Dump.zip"
    
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
        zip_file.extractall(unzip_dir)
    os.remove(zip_fileloc)

    print("Finished: file bx is downloaded to the directory: %s" % 
          (os.path.join(samples_dir, unzip_dir)))

def download_jester_data(label : str,
                         dir   : str = None):

    # check if the label is allowed
    if label not in __jester_label__:
        raise ValueError("label must be in [%s]." % (", ".join(__jester_label__)))
    
    # set directory name and create directory if not exist
    if dir is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, "sample_data")
    else:
        samples_dir = dir
    
    # make directory of unzip_dir
    unzip_dir = os.path.join(samples_dir, "jester-%s" % label)
    Path(unzip_dir).mkdir(parents=True, exist_ok=True)

    # set file name to download data
    zip_filename = ("jester-%s.zip" % label)
    zip_fileurl = ("https://goldberg.berkeley.edu/jester-data/jester-data-%s.zip" % label)
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
        zip_file.extractall(unzip_dir)
    os.remove(zip_fileloc)

    print("Finished: file %s is downloaded to the directory: %s" % 
          ("jester-%s" % label, os.path.join(samples_dir, unzip_dir)))
