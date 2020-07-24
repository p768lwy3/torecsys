r"""torecsys.data.sampledata is a sub module to download and load sample dataset, like movielens
"""
import os
from typing import List

__ml_size__ = ["20m", "latest-small", "latest", "100k", "1m", "10m"]
__jester_label__ = ["1", "2", "3"]


def get_downloaded_data(dir: str = None) -> List[str]:
    r"""Get list of downloaded dataset
    
    Args:
        dir (str, optional): String of directory of downloaded data. 
            Defaults to None.
    
    Returns:
        List[str]: List of string of downloaded dataset, 
            i.e. subdirectory names in input directory.
    """
    # set directory name of the downloaded data
    if dir is None:
        script_dir = os.path.dirname(__file__)
        dir = os.path.join(script_dir, "sample_data")
    else:
        pass

    # scan the directory and check if it is a directory
    files = [f.name for f in os.scandir(dir) if f.is_dir()] if os.path.isdir(dir) else []

    return files


def check_downloaded(dataset: str,
                     dir: str = None) -> bool:
    r"""Check whether dataset is downloaded
    
    Args:
        dataset (str): String of dataset's name, e.g. ml-100k, bx
        dir (str, optional): String of directory of downloaded data. 
            Defaults to None.
    
    Returns:
        bool: Boolean flag to show if the dataset is downloaded, 
            i.e. name of dataset is in the list of subdirectory in input directory.
    """
    # get list of subdirectory in input directory
    dirs = get_downloaded_data(dir=dir)

    # check if dataset is in the list of subdirectory in input directory
    if dataset in dirs:
        return True
    else:
        return False


from .download_data import download_ml_data, download_bx_data, download_jester_data
from .load_data import load_ml_data, load_bx_data
