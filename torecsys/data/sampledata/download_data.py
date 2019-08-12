from . import __ml_size__
import math
import os
import requests
from tqdm.autonotebook import tqdm
import zipfile

def download_ml_data(size : str):
    r"""Download movielens data from grouplens to directory ../sample_data/,
    and the source url of movielens is : https://grouplens.org/datasets/movielens/
    
    Args:
        size (str): movielens dataset size, allows: 20m, latest-small, latest, 100k, 1m, 10m
    
    Raises:
        ValueError: when size is not in allowed values 
        RuntimeError: when download face trouble 
    """
    # check if the size is allowed
    if size not in __ml_size__:
        raise ValueError("size must be in [%s]." % (", ".join(__ml_size__)))
    
    # file name and file path
    script_dir = os.path.dirname(__file__)
    samples_dir = os.path.join(script_dir, "sample_data")
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
