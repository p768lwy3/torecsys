from torecsys.utils.logging.decorator import to_be_tested

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import torch
import torch.utils.data
from typing import List


class NdarrayToDataset(torch.utils.data.Dataset):
    r"""Conver np.ndarray to torch.utils.data.Dataset per row
    """
    def __init__(self,
                 ndarray: np.ndarray):
        r"""initialize NdarrayToDataset
        
        Args:
            ndarray (np.ndarray): dataset of ndarray
        """
        super(NdarrayToDataset, self).__init__()
        self.data = np.ndarray
    
    def __len__(self) -> int:
        r"""Return number of rows of dataset
        
        Returns:
            int: number of rows of dataset
        """
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> List[list]:
        r"""Get a row in dataset
        
        Args:
            idx (int): index of row
        
        Returns:
            List[list]: List of lists which is storing features of a field in dataset
        """
        row = self.data[idx].tolist()
        return [[v] for v in row]


class DataFrameToDataset(torch.utils.data.Dataset):
    r"""Convert pd.DataFrame to torch.utils.data.Dataset per row
    """
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 columns  : List[str]):
        r"""initialize DataFrameToDataset
        
        Args:
            dataframe (pd.DataFrame): dataset of DataFrame
            columns (List[str]): column names of fields
        """
        # store variables to get row of data
        super(DataFrameToDataset, self).__init__()
        self.data = dataframe
        self.columns = columns

    def __len__(self) -> int:
        r"""Return number of rows of dataset
        
        Returns:
            int: number of rows of dataset
        """
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> List[list]:
        r"""Get a row in dataset
        
        Args:
            idx (int): index of row
        
        Returns:
            List[list]: List of lists which is storing features of a field in dataset
        """
        row = self.data.iloc[idx][self.columns].tolist()
        return [[v] for v in row]


@to_be_tested
class SqlalchemyToDataset(torch.utils.data.Dataset):
    r"""[to be tested] Convert a SQL query to torch.utils.data.Dataset
    """
    def __init__(self,
                 sql_engine: Engine,
                 sql_query : str,
                 columns   : List[str]):
        r"""initialize SqlalchemyToDataset
        
        Args:
            sql_engine (sqlalchemy.engine.base.Engine): connection of sql server
            sql_query (str): query of sql
            columns (List[str]): column names of fields
        """
        # read sql to dataframe
        super(DataFrameToDataset, self).__init__()
        self.data = pd.reqd_sql(sql_query, sql_engine)
        self.columns = columns

    def __len__(self) -> int:
        r"""Return number of rows of dataset
        
        Returns:
            int: number of rows of dataset
        """
        return self.data.shape[0]
    
    def __getitem__(self, idx: int) -> List[list]:
        r"""Get a row in dataset
        
        Args:
            idx (int): index of row
        
        Returns:
            List[list]: List of lists which is storing features of a field in dataset
        """
        row = self.data.iloc[idx][self.columns].tolist()
        return [[v] for v in row]


@to_be_tested
class CooToDataset(torch.utils.data.Dataset):
    r"""[to be developed] Conver scipy.sparse.coo_matrix to torch.utils.data.Dataset
    
    Reference:
        https://pytorch.org/docs/stable/sparse.html
    """
    def __init__(self, 
                 coo_matrix: sparse.coo_matrix):
        raise NotImplementedError("not yet started to development")
