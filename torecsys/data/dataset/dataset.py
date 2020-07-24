from typing import Dict, List, Union, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sparse
import torch.utils.data
from sqlalchemy.engine import Engine

from torecsys.utils.decorator import to_be_tested


class NdarrayToDataset(torch.utils.data.Dataset):
    r"""Convert np.ndarray to torch.utils.data.Dataset per row
    """

    def __init__(self,
                 ndarray: np.ndarray):
        r"""initialize NdarrayToDataset
        
        Args:
            ndarray (np.ndarray): dataset of ndarray
        """
        super(NdarrayToDataset, self).__init__()
        self.ndarray = ndarray
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
                 columns: List[str],
                 names: Tuple[str] = None,
                 use_dict: bool = True):
        r"""initialize DataFrameToDataset
        
        Args:
            dataframe (pd.DataFrame): dataset of DataFrame
            columns (List[str]): column names of fields
            use_dict (bool, optional): boolean flag to control using dictionary or list to response. Default to True.
        """
        # refer to parent class
        super(DataFrameToDataset, self).__init__()

        # bind dataframe, columns and use_dict to data, columns and use_dict
        self.columns = columns
        self.data = dataframe
        self.names = names
        self.use_dict = use_dict

    def __len__(self) -> int:
        r"""Return number of rows of dataset
        
        Returns:
            int: number of rows of dataset
        """
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> Union[Dict[str, list], List[list]]:
        r"""Get a row in dataset
        
        Args:
            idx (int): index of row
        
        Returns:
            List[list]: List of lists which is storing features of a field in dataset
        """
        # get data from self.data by field names in self.columns
        rows = self.data.iloc[idx][self.columns].tolist()

        # transform to dictionary or list and return
        if self.use_dict:
            return {k: [v] if not isinstance(v, list) else v for k, v in zip(self.columns, rows)}
        else:
            return [[v] if not isinstance(v, list) else v for v in rows]


@to_be_tested
class SqlalchemyToDataset(torch.utils.data.Dataset):
    r"""[to be tested] Convert a SQL query to torch.utils.data.Dataset
    """

    def __init__(self,
                 sql_engine: Engine,
                 sql_query: str,
                 columns: List[str]):
        r"""initialize SqlalchemyToDataset
        
        Args:
            sql_engine (sqlalchemy.engine.base.Engine): connection of sql server
            sql_query (str): query of sql
            columns (List[str]): column names of fields
        """
        # read sql to dataframe
        super(SqlalchemyToDataset, self).__init__()
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
    r"""[to be developed] Convert scipy.sparse.coo_matrix to torch.utils.data.Dataset
    
    Reference:
        https://pytorch.org/docs/stable/sparse.html
    """

    def __init__(self,
                 coo_matrix: sparse.coo_matrix):
        self.coo_matrix = coo_matrix
        raise NotImplementedError("not yet started to development")
