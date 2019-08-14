r"""torecsys.data.sampledata is a sub module to download and load sample dataset, like movielens
"""

__ml_size__ = ["20m", "latest-small", "latest", "100k", "1m", "10m"]

from .download_data import download_ml_data
from .load_data import load_ml_data
