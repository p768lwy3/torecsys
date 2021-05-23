"""

"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch.utils.data


class DataFrameToDataset(torch.utils.data.Dataset):
    """
    Convert pd.DataFrame to torch.utils.data.Dataset per row
    """

    def __init__(self,
                 dataframe: pd.DataFrame,
                 columns: List[str],
                 use_dict: bool = True):
        """
        Initialize DataFrameToDataset
        
        Args:
            dataframe (pd.DataFrame): dataset of DataFrame
            columns (List[str]): column names of fields
            use_dict (bool, optional): boolean flag to control using dictionary or list to response. Default to True.
        """
        super().__init__()

        self.columns = columns
        self.data = dataframe
        self.use_dict = use_dict

    def __len__(self) -> int:
        """
        Return size of dataset
        
        Returns:
            int: size of dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Union[Dict[str, list], List[list]]:
        """
        Get a row in dataset
        
        Args:
            idx (int): index of row
        
        Returns:
            Union[Dict[str, list], List[list]]: Dict or list of lists which is storing features of a field in dataset
        """
        rows = self.data.iloc[idx][self.columns].tolist()

        if self.use_dict:
            return {k: [v] if not isinstance(v, list) else v for k, v in zip(self.columns, rows)}
        else:
            return [[v] if not isinstance(v, list) else v for v in rows]


class NdarrayToDataset(torch.utils.data.Dataset):
    """
    Convert np.ndarray to torch.utils.data.Dataset per row
    """

    def __init__(self,
                 ndarray: np.ndarray):
        """
        Initialize NdarrayToDataset

        Args:
            ndarray (np.ndarray): dataset of ndarray
        """
        super().__init__()

        self.data = ndarray

    def __len__(self) -> int:
        """
        Return size of dataset

        Returns:
            int: size of dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> List[list]:
        """
        Get a row in dataset

        Args:
            idx (int): index of row

        Returns:
            List[list]: List of lists which is storing features of a field in dataset
        """
        return [[v] if not isinstance(v, list) else v for v in self.data[idx].tolist()]
