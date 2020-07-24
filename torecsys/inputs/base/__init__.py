r"""torecsys.inputs.base is a sub-module of base inputs class.
"""

from collections import namedtuple
from typing import List, Union

import torch.nn as nn


class Inputs(nn.Module):
    r"""General Input class.
    """

    def __init__(self):
        # refer to parent class
        super(Inputs, self).__init__()

    def __len__(self) -> int:
        r"""Return outputs size.
        
        Returns:
            int: Size of embedding tensor, or Number of inputs' fields.
        """
        return self.length

    def set_schema(self, inputs: Union[str, List[str]]):
        r"""Initialize input layer's schema.
        
        Args:
            inputs (Union[str, List[str]]): String or list of strings of inputs' field names.
        """
        # convert string to list of string
        if isinstance(inputs, str):
            inputs = [inputs]

        # create a namedtuple of schema
        schema = namedtuple("Schema", ["inputs"])

        # initialize self.schema with the namedtuple
        self.schema = schema(inputs=inputs)


# from .audio_inp import AudioInputs
from .concat_inputs import ConcatInputs
from .image_inp import ImageInputs
# from .images_list_inp import ImageListInputs
from .list_indices_emb import ListIndicesEmbedding
from .multi_indices_emb import MultiIndicesEmbedding
from .multi_indices_field_aware_emb import MultiIndicesFieldAwareEmbedding
from .pretrained_image_inp import PretrainedImageInputs
# from .pretrained_text_inp import PretrainedTextInputs
from .sequence_indices_emb import SequenceIndicesEmbedding
from .single_index_emb import SingleIndexEmbedding
from .stacked_inp import StackedInputs
# from .text_inp import TextInputs
# from .timeseries_inp import TimeseriesInputs
# from .timestamp_inp import TimestampInputs
from .value_inp import ValueInputs
