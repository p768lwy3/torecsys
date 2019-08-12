from . import __ml_size__
import pandas as pd
import os

def load_ml_data(size : str) -> pd.DataFrame:
    r"""Load movielens dataset from ml-[size]/ratings.csv file in ../sample_data to pandas.DataFrame.
    
    Args:
        size (str): movielens dataset size, allows: 20m, latest-small, latest, 100k, 1m, 10m
    
    Raises:
        ValueError: when size is not in allowed values 
    
    Returns:
        pd.DataFrame: movielens dataset dataframe, with columns = [userId, movieId, rating, timestamp]
    """
    # check if the size is allowed
    if size not in __ml_size__:
        raise ValueError("size must be in [%s]." % (", ".join(__ml_size__)))

    # file name and file path
    script_dir = os.path.dirname(__file__)
    data_filepath = os.path.join(script_dir, ("sample_data/ml-%s/ratings.csv" % size))

    df = pd.read_csv(data_filepath)
    return df
