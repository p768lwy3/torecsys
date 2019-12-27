from . import __ml_size__
import pandas as pd
import os

def load_ml_data(size : str,
                 dir  : str = None) -> pd.DataFrame:
    r"""Load movielens dataset from ml-[size]/ratings.csv file in ../sample_data to pandas.DataFrame.
    
    Args:
        size (str): Movielens dataset size, allows: 20m, latest-small, latest, 100k, 1m, 10m
        dir (str, optional): Directory to save downloaded data. Default to None.

    Raises:
        ValueError: when size is not in allowed values 
    
    Returns:
        pd.DataFrame: movielens dataset dataframe, with columns = [userId, movieId, rating, timestamp]
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
