"""
torecsys.data.sample_data is a sub model to download and load sample dataset, like movielens
"""

import nt
import os
from typing import List

__ml_size__ = ['20m', 'latest-small', 'latest', '100k', '1m', '10m']
__jester_label__ = ['1', '2', '3']


def get_downloaded_data(directory: str = None) -> List[str]:
    """
    Get list of downloaded dataset
    
    Args:
        directory (str, optional): String of directory of downloaded data.
            Defaults to None.
    
    Returns:
        List[str]: List of string of downloaded dataset, 
            i.e. subdirectory names in input directory.
    """
    # set directory name of the downloaded data
    if directory is None:
        script_dir = os.path.dirname(__file__)
        directory = os.path.join(script_dir, 'sample_data')
    else:
        pass

    # scan the directory and check if it is a directory
    f: nt.DirEntry
    files = [f.name for f in os.scandir(directory) if f.is_dir()] if os.path.isdir(directory) else []

    return files


def check_downloaded(dataset: str,
                     directory: str = None) -> bool:
    """
    Check whether dataset is downloaded
    
    Args:
        dataset (str): String of dataset's name, e.g. ml-100k, bx
        directory (str, optional): String of directory of downloaded data.
            Defaults to None.
    
    Returns:
        bool: Boolean flag to show if the dataset is downloaded, 
            i.e. name of dataset is in the list of subdirectory in input directory.
    """
    return True if dataset in get_downloaded_data(directory=directory) else False


from torecsys.data.sample_data.download_data import download_ml_data, download_bx_data, download_jester_data
from torecsys.data.sample_data.load_data import load_ml_data, load_bx_data
