from . import __ml_size__, check_downloaded
from .download_data import download_ml_data, download_bx_data, download_data
import pandas as pd
import os
from typing import List

def load_ml_data(size  : str,
                 dir   : str = None,
                 force : bool = False) -> List[pd.DataFrame]:
    r"""Load movielens dataset from ./sample_data/ml-[size] to pd.DataFrame.
    
    Args:
        size (str): Movielens dataset size, allows: 20m, latest-small, latest, 100k, 1m, 10m
        dir (str, optional): Directory to save downloaded data. 
            Defaults to None.
        force (bool, optional): Download dataset if it is not found in directory when force is True.
            Defaults to False.

    Raises:
        ValueError: when size is not in allowed values
        ValueError: when dataset cannot be found in given directory
    
    Returns:
        List[pd.DataFrame]: movielens dataset
    """
    # check if the size is allowed
    if size not in __ml_size__:
        raise ValueError("size must be in [%s]." % (", ".join(__ml_size__)))

    # set directory name of the downloaded data
    if dir is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, "sample_data")
    else:
        samples_dir = dir

    # check if the dataset is in the directory
    is_exist = check_downloaded("-".join(["ml", size]))
    if force is False and is_exist is False:
        raise ValueError("dataset haven't been found in %s" % samples_dir)
    elif force is True and is_exist is False:
        print("dataset haven't been found in %s." % samples_dir)
        download_data("-".join(["ml", size]), dir=samples_dir)
    
    # set file path to load the data
    links_path   = os.path.join(samples_dir, ("ml-%s/links.csv" % size))
    movies_path  = os.path.join(samples_dir, ("ml-%s/movies.csv" % size))
    ratings_path = os.path.join(samples_dir, ("ml-%s/ratings.csv" % size))
    tags_path    = os.path.join(samples_dir, ("ml-%s/tags.csv" % size))

    # read csv file as DataFrame
    links_df = pd.read_csv(links_path)
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    tags_df = pd.read_csv(tags_path)

    return links_df, movies_df, ratings_df, tags_df

def load_bx_data(dir   : str = None,
                 force : bool = False) -> List[pd.DataFrame]:
    r"""Load Book-Crossing dataset from ./sample_data/bx to pd.DataFrame.
    
    Args:
        dir (str, optional): Directory to save downloaded data. 
            Defaults to None.
        force (bool, optional): Download dataset if it is not found in directory when force is True.
            Defaults to False.

    Raises:
        ValueError: when dataset cannot be found in given directory
    
    Returns:
        List[pd.DataFrame]: Book-Crossing dataset
    """
    # set directory name of the downloaded data
    if dir is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, "sample_data")
    else:
        samples_dir = dir

    # check if the dataset is in the directory
    is_exist = check_downloaded("bx")
    if force is False and is_exist is False:
        raise ValueError("dataset haven't been found in %s" % samples_dir)
    elif force is True and is_exist is False:
        print("dataset haven't been found in %s." % samples_dir)
        download_data("bx", dir=samples_dir)
    
    # set file path to load the data
    ratings_path = os.path.join(samples_dir, "bx/BX-Book-Ratings.csv")
    books_path = os.path.join(samples_dir, "bx/BX-Books.csv")
    users_path = os.path.join(samples_dir, "bx/BX-Users.csv")

    # read csv file as DataFrame
    ratings_df = pd.read_csv(ratings_path, 
        sep=";", engine="python", error_bad_lines=False, warn_bad_lines=False)
    books_df = pd.read_csv(books_path, sep=";", 
        engine="python", error_bad_lines=False, warn_bad_lines=False)
    users_df = pd.read_csv(users_path, sep=";", 
        engine="python", error_bad_lines=False, warn_bad_lines=False)

    return ratings_df, books_df, users_df

def load_jester_data(label : str,
                     dir   : str = None,
                     force : bool = False) -> List[pd.DataFrame]:
    return 
