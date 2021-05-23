from typing import Iterable, Union

import torch

Ints = Union[int, Iterable['Ints']]

Strings = Union[str, Iterable['Strings']]

Tensors = Union[torch.Tensor, Iterable['Tensors']]
