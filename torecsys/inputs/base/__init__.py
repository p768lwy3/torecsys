"""
torecsys.inputs.base is a sub-model of base inputs class
"""
from abc import ABC
from collections import namedtuple
from typing import List, Union

import torch.nn as nn


class BaseInput(nn.Module, ABC):
    """
    General Input class.
    """

    def __init__(self):
        """
        Initializer of the inputs
        """
        super().__init__()
        self.schema = None

    def __len__(self) -> int:
        """
        Return outputs size
        
        Returns:
            int: size of embedding tensor, or number of inputs' fields
        """
        return self.length

    def set_schema(self, inputs: Union[str, List[str]], **kwargs):
        """
        Initialize input layer's schema
        
        Args:
            inputs (Union[str, List[str]]): string or list of strings of inputs' field names
        """
        # convert string to list of string
        if isinstance(inputs, str):
            inputs = [inputs]

        schema = namedtuple('Schema', ['inputs'])

        self.schema = schema(inputs=inputs)


__all__ = [
    'ConcatInput',
    'ImageInput',
    'BaseInput',
    'ListIndicesEmbedding',
    'MultiIndicesEmbedding',
    'MultiIndicesFieldAwareEmbedding',
    'PretrainedImageInput',
    'SequenceIndicesEmbedding',
    'SingleIndexEmbedding',
    'StackedInput',
    'ValueInput'
]

from torecsys.inputs.base.concat_inputs import ConcatInput
from torecsys.inputs.base.image_inp import ImageInput
from torecsys.inputs.base.list_indices_emb import ListIndicesEmbedding
from torecsys.inputs.base.multi_indices_emb import MultiIndicesEmbedding
from torecsys.inputs.base.multi_indices_field_aware_emb import MultiIndicesFieldAwareEmbedding
from torecsys.inputs.base.pretrained_image_inp import PretrainedImageInput
from torecsys.inputs.base.sequence_indices_emb import SequenceIndicesEmbedding
from torecsys.inputs.base.single_index_emb import SingleIndexEmbedding
from torecsys.inputs.base.stacked_inp import StackedInput
from torecsys.inputs.base.value_inp import ValueInput
