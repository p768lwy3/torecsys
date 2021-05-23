"""

"""
import collections.abc
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import requests
import torch
import torch.nn.utils.rnn as rnn_utils
import torchvision.transforms as transforms
from PIL import Image
from texttable import Texttable

from torecsys.data.dataloader.fields import IndexField


class CollateFunction(object):
    CollateFunction = TypeVar('CollateFunction')

    FIELD_TYPE_ENUM = ['values', 'indices', 'images']

    def __init__(self,
                 schema: Dict[str, Any],
                 device: str = 'cpu',
                 kwargs: Dict[str, Any] = None):
        """
        Initializer of Collator

        Args:
            schema (Dict[str, Any]):
            device (str, optional):
            kwargs (Dict[str, Any], optional):
        """
        if kwargs is None:
            kwargs = {}

        if not isinstance(schema, dict):
            raise TypeError(f'Type of schema {type(schema).__name__} is not allowed.')

        self.schema = schema

        if not isinstance(device, str):
            raise TypeError(f'Type of device {type(device).__name__} is not allowed')

        self.device = device

        if not isinstance(kwargs, dict):
            raise TypeError(f'Type of kwargs {type(kwargs).__name__} is not allowed')

        self.kwargs = kwargs

    def _collate_values(self, inp_values: List[list]) -> torch.Tensor:
        """
        Convert inp_values from list to torch.tensor
        
        Args:
            inp_values (List[list]): list of batch values
        
        Returns:
            T, data_type=torch.float32: torch tensor of batch values
        """
        return torch.Tensor(inp_values).to(self.device)

    def _collate_indices(self,
                         inp_values: Union[List[int], List[List[int]]],
                         mapping: IndexField = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert inp_values from list to torch.tensor
        
        Args:
            inp_values (List[int]): list of batch values
            mapping (IndexField, optional): indexField to map index to token. defaults to None
        
        Returns:
            T, data_type=torch.int32: torch tensor of batch values
        """
        inp_values = mapping.fit_predict(inp_values) if mapping is not None else inp_values
        max_len = max([len(lst) for lst in inp_values])

        if max_len == 1:
            return torch.Tensor(inp_values).long().to(self.device)
        else:
            inp_values_index = zip(inp_values, range(len(inp_values)))
            inp_values_index = sorted(inp_values_index, key=lambda x: len(x[0]), reverse=True)
            perm_tuple = [(c, s) for c, s in inp_values_index]

            perm_tensors = [torch.Tensor(lst) for lst, _ in perm_tuple]
            perm_lengths = torch.Tensor([len(t) for t in perm_tensors])
            perm_idx = [idx for _, idx in perm_tuple]

            padded_t = rnn_utils.pad_sequence(perm_tensors, batch_first=True, padding_value=0)
            desort_idx = list(sorted(range(len(perm_idx)), key=perm_idx.__getitem__))
            desort_t = padded_t[desort_idx].long().to(self.device)
            desort_len = perm_lengths[desort_idx].long().to(self.device)
            return desort_t, desort_len

    def _collate_images(self,
                        inp_values: List[str],
                        input_type: str,
                        transforms_method: transforms = transforms.ToTensor(),
                        file_root: str = None) -> torch.Tensor:
        """
        Load image with inp_values and convert them to tensor

        Args:
            inp_values (List[str]): list of batch values
            input_type (str): type of inp_values
            transforms_method (transforms, optional): transforms method from torchvision.
                defaults to transforms.ToTensor()
            file_root (str, optional): string of files' root. defaults to None

        Returns:
            T: torch tensor of batch values
        """
        if file_root is not None:
            inp_values = [file_root + inp for inp in inp_values]

        if not isinstance(input_type, str):
            raise TypeError(f'Type of input_type {type(input_type).__name__} is not allowed')

        input_types = ['file', 'url']
        if input_type not in input_types:
            raise AssertionError(f'input_type {input_type} is not allowed, only allow {input_types}')

        load_method = Image.open if input_type == 'file' else lambda url: Image.open(BytesIO(requests.get(url).content))
        images = torch.stack([transforms_method(load_method(img)) for img in inp_values])
        return images.to(self.device)

    def _collate(self,
                 inp_values: List[list],
                 collate_type: str,
                 **kwargs: dict) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Convert batch of input to tensor
        
        Args:
            inp_values (List[list]): list of batch values
            collate_type (str): type of collation
            **kwargs (dict): see below
        
        Returns:
            Union[T, Tuple[T, T]]: torch tensor of batch values
        """
        if not isinstance(collate_type, str):
            raise TypeError(f'Type of collate_type {type(collate_type).__name__} is not allowed')

        if collate_type not in self.FIELD_TYPE_ENUM:
            raise AssertionError(f'collate_type {collate_type} is not allowed, only accept: {self.FIELD_TYPE_ENUM}')

        _collate_method = getattr(self, f'_collate_{collate_type}')
        return _collate_method(inp_values, **kwargs)

    def to_tensor(self, batch_data: List[Dict[str, list]]) \
            -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """

        Args:
            batch_data:

        Returns:

        """
        outputs = {}

        for field_out_key, field in self.schema.items():
            if isinstance(field, str):
                field_type = field
            elif isinstance(field, collections.abc.Sequence):
                field_inp_key, field_type = field
            else:
                raise ValueError(f'type of field {type(field)} is not acceptable')

            inp_values = [b[field_inp_key] for b in batch_data]
            kwargs = self.kwargs.get(field_out_key, {})
            outputs[field_out_key] = self._collate(inp_values, field_type, **kwargs)

        return outputs

    def summary(self,
                deco: int = Texttable.BORDER,
                cols_align: Optional[List[str]] = None,
                cols_valign: Optional[List[str]] = None) -> CollateFunction:
        """
        Get summary of trainer

        Args:
            deco (int): border of texttable
            cols_align (List[str], optional): list of string of columns' align
            cols_valign (List[str], optional): list of string of columns' valign
        
        Returns:
            torecsys.data.dataloader.CollateFunction: self
        """
        if cols_align is None:
            cols_align = ['l', 'l', 'l']

        if cols_valign is None:
            cols_valign = ['t', 't', 't']

        t = Texttable()
        t.set_deco(deco)
        t.set_cols_align(cols_align)
        t.set_cols_valign(cols_valign)
        t.add_rows(
            [['Field Name: ', 'Field Type: ', 'Arguments: ']] +
            [[k, v, ', '.join(self.kwargs.get(k, {}).keys())]
             for k, v in self.schema.items()]
        )

        print(t.draw())

        return self
