from io import BytesIO
from PIL import Image
import os
import requests
from texttable import Texttable
import torch
import torch.nn.utils.rnn as rnn_utils
import torchvision
import torchvision.transforms as transforms
from torecsys.data.dataloader import IndexField
from typing import Dict, List, Tuple, TypeVar, Union
import warnings

__field_type__ = [
    "values", "indices", "images"
]

class DataloaderCollator(object):
    def __init__(self, 
                 schema : dict,
                 device : str  = "cpu",
                 kwargs : dict = {}):
        
        if not isinstance(schema, dict):
            raise TypeError(f"{type(schema).__name__} not allowed.")

        self.schema = schema

        if not isinstance(device, str):
            raise TypeError(f"{type(device).__name__} not allowed.")
        
        self.device = device

        if not isinstance(kwargs, dict):
            raise TypeError(f"{type(kwargs).__name__} not allowed.")
        
        self.kwargs = kwargs
    
    def _collate_values(self, inp_values: List[list]) -> torch.Tensor:
        r"""Convert inp_values from list to torch.tensor.
        
        Args:
            inp_values (List[list]): list of batch values.
        
        Returns:
            T, dtype=torch.float32: torch tensor of batch values.
        """
        return torch.Tensor(inp_values).to(self.device)
    
    def _collate_indices(self, 
                         inp_values : List[list], 
                         mapping    : IndexField = None) -> torch.Tensor:
        r"""Convert inp_values from list to torch.tensor.
        
        Args:
            inp_values (List[list]): list of batch values.
            mapping (IndexField, optional): IndexField to map index to token. 
                Defaults to None.
        
        Returns:
            T, dtype=torch.int32: torch tensor of batch values.
        """
        inp_values = mapping.fit_predict(inp_values) if mapping is not None \
            else inp_values
        
        max_len = max([len(lst) for lst in inp_values])

        if max_len == 1:
            # Convert to torch.tensor directly
            return torch.Tensor(inp_values).long().to(self.device)
        
        else:
            warnings.warn("Not checked.")

            # Sort inp_values in descending order
            inp_values_index = zip(inp_values, range(len(inp_values)))
            inp_values_index = sorted(inp_values_index, key=lambda x: len(x[0]), reverse=True)
            perm_tuple = [(c, s) for c, s in inp_values_index]

            # Convert lists inside list to tensor
            perm_tensors = [torch.Tensor(lst) for lst, _ in perm_tuple]
            perm_lengths = torch.Tensor([len(t) for t in perm_tensors])
            perm_idx = [idx for _, idx in perm_tuple]

            # Pad sequences
            padded_t = rnn_utils.pad_sequence(perm_tensors, batch_first=True, padding_value=0)

            # Desort padded tensor
            desort_idx = list(sorted(range(len(perm_idx)), key=perm_idx.__getitem__))
            desorted_t = padded_t[desort_idx].long().to(self.device)
            desorted_len = perm_lengths[desort_idx].long().to(self.device)

            return desorted_t, desorted_len
    
    def _collate_images(self,
                        inp_values        : List[list],
                        input_type        : str,
                        transforms_method : transforms = transforms.ToTensor(),
                        file_root         : str = None) -> torch.Tensor:
        r"""Load image with inp_values and convert them to tensor
        
        Args:
            batch_inp (List[list]): list of batch values.
            input_type (str): Type of inp_values.
            transforms_method (transforms, optional): Trasforms method from torchvision. 
                Defaults to transforms.ToTensor().
            file_root (str, optional): String of files' root.
                Defaults to None.
        
        Returns:
            T: torch tensor of batch values.
        """
        warnings.warn("Not checked.")

        if file_root is not None:
            inp_values = [file_root + inp for inp in inp_values]
        
        if not isinstance(input_type, str):
            raise TypeError(f"{type(input_type).__name__} not allowed.")
        
        if input_type not in ["file", "url"]:
            raise AssertionError(f"{input_type} not allowed.")

        load_method = Image.open if input_type == "file" \
            else lambda url: Image.open(BytesIO(requests.get(url).content))
        
        images = [load_method(img) for img in inp_values]
        images = [transforms_method(img) for img in images]
        images = torch.stack(images)

        return images

    def _collate(self,
                 inp_values   : List[list],
                 collate_type : str,
                 kwargs       : dict = None) \
            -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Convert batch of input to tensor.
        
        Args:
            inp_values (List[list]): list of batch values.
            collate_type (str): Type of collation.
        
        Returns:
            Union[T, Tuple[T, T]]: torch tensor of batch values.
        """
        if not isinstance(collate_type, str):
            raise TypeError(f"{type(collate_type).__name__} not allowed.")
        
        if collate_type not in __field_type__:
            raise AssertionError(f"{collate_type} not allowed.")

        _collate_func = "_collate_" + collate_type
        _collate_method = getattr(self, _collate_func)
        
        return _collate_method(inp_values, **kwargs)
    
    def to_tensor(self, 
                  batch_data: List[Dict[str, list]]) -> Dict[str, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        
        outputs = dict()

        for field_name, field_type in self.schema.items():
            # Get inputs and kwargs by input_key in self.schema.items()
            inp_values = [b[field_name] for b in batch_data]
            kwargs = self.kwargs.get(field_name, {})

            # Return tensor by _collate
            outputs[field_name] = self._collate(inp_values, field_type, kwargs)
        
        return outputs

    def summary(self,
                deco        : int = Texttable.BORDER,
                cols_align  : List[str] = ["l", "l", "l"],
                cols_valign : List[str] = ["t", "t", "t"]) -> TypeVar("DataloaderCollator"):
        r"""Get summary of trainer.

        Args:
            deco (int): Border of texttable
            cols_align (List[str]): List of string of columns' align
            cols_valign (List[str]): List of string of columns' valign
        
        Returns:
            torecsys.data.dataloader.DataloaderCollator: self
        """
         # Create and configurate Texttable
        t = Texttable()
        t.set_deco(deco)
        t.set_cols_align(cols_align)
        t.set_cols_valign(cols_valign)
        
        # Append data to texttable
        t.add_rows(
            [["Field Name: ", "Field Type: ", "Arguments: "]] + \
            [[k, v, ", ".join(self.kwargs.get(k, {}).keys())] \
                for k, v in self.schema.items()]
        )

        # Print summary with texttable
        print(t.draw())

        return self
