"""

"""

import os
from typing import Tuple

import pandas as pd

from torecsys.data.sample_data import __ml_size__, check_downloaded
from torecsys.data.sample_data.download_data import download_data


def load_ml_data(size: str,
                 directory: str = None,
                 force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load movielens dataset from directory/ml-[size] to pd.DataFrame.
    Args:
        size (str): Movielens dataset size, allows: 20m, latest-small, latest, 100k, 1m, 10m
        directory (str, optional): Directory to save downloaded data. 
            Defaults to None.
        force (bool, optional): Download dataset if it is not found in directory when force is True.
            Defaults to False.

    Raises:
        ValueError: when size is not in allowed values
        ValueError: when dataset cannot be found in given directory
    
    Returns:
        Tuple[df, df, df, df]: Tuple of movielens dataset dataframe
    """
    # check if the size is allowed
    if size not in __ml_size__:
        raise ValueError("size must be in [%s]." % (", ".join(__ml_size__)))

    # set directory name of the downloaded data
    if directory is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, "sample_data")
    else:
        samples_dir = directory

    # check if the dataset is in the directory
    is_exist = check_downloaded("-".join(["ml", size]), directory)
    if force is False and is_exist is False:
        raise ValueError("dataset haven't been found in %s" % samples_dir)
    elif force is True and is_exist is False:
        print("dataset haven't been found in %s." % samples_dir)
        download_data("-".join(["ml", size]), dir=samples_dir)

    # set file path to load the data
    links_path = os.path.join(samples_dir, ("ml-%s/links.csv" % size))
    movies_path = os.path.join(samples_dir, ("ml-%s/movies.csv" % size))
    ratings_path = os.path.join(samples_dir, ("ml-%s/ratings.csv" % size))
    tags_path = os.path.join(samples_dir, ("ml-%s/tags.csv" % size))

    # read csv file as DataFrame
    links_df = pd.read_csv(links_path)
    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    tags_df = pd.read_csv(tags_path)

    return links_df, movies_df, ratings_df, tags_df


def load_criteo_data(directory: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load criteo dataset from directory/dac to pd.DataFrame.
    
    Args:
        directory (str, optional): Directory to save downloaded data. Default to None.
    
    Returns:
        Tuple[df, df]: Tuple of criteo click-through-rate dataframe, with 39 columns.
    """
    # set directory name of the downloaded data:
    if directory is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, "sample_data")
    else:
        samples_dir = directory

    # set file path to dac
    train_path = os.path.join(samples_dir, "dac/train.txt")
    test_path = os.path.join(samples_dir, "dac/tests.txt")

    columns = ["col_%s" % str(i) for i in range(0, 40)]

    with open(train_path, "r") as train_txt:
        train_file = pd.read_csv(train_txt, sep="\t", names=columns)

    with open(test_path, "r") as test_txt:
        test_file = pd.read_csv(test_txt, sep="\t", names=columns)

    return train_file, test_file


def load_bx_data(directory: str = None,
                 force: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Book-Crossing dataset from ./sample_data/bx to pd.DataFrame.
    
    Args:
        directory (str, optional): Directory to save downloaded data.
            Defaults to None.
        force (bool, optional): Download dataset if it is not found in directory when force is True.
            Defaults to False.

    Raises:
        ValueError: when dataset cannot be found in given directory
    
    Returns:
        Tuple[df, df, df]: Book-Crossing dataset
    """
    # set directory name of the downloaded data
    if directory is None:
        script_dir = os.path.dirname(__file__)
        samples_dir = os.path.join(script_dir, "sample_data")
    else:
        samples_dir = directory

    # check if the dataset is in the directory
    is_exist = check_downloaded("bx", directory)
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
