r"""torecsys.inputs.base is a sub-module of base class to be called for inputs
"""

import torch.nn as nn

class _Inputs(nn.Module):
    r"""Base Modle Class of inputs or embedding field"""
    def __init__(self):
        super(_Inputs, self).__init__()
    
    def __len__(self) -> int:
        r"""Return output size of _Input field
        
        Returns:
            int: embedding size for embedding field, or number of fields for input
        """
        return self.length


# from .audio_inp import AudioInputs
from .concat_inputs import ConcatInputs
from .image_inp import ImageInputs
# from .images_list_inp import ImageListInputs
from .list_indices_emb import ListIndicesEmbedding
from .multi_indices_emb import MultiIndicesEmbedding
from .multi_indices_field_aware_emb import MultiIndicesFieldAwareEmbedding
from .pretrained_image_inp import PretrainedImageInputs
# from .pretrained_text_inp import PretrainedTextInputs
from .sequence_index_emb import SequenceIndexEmbedding
from .single_index_emb import SingleIndexEmbedding
from .stacked_inp import StackedInputs
# from .text_inp import TextInputs
# from .timeseries_inp import TimeseriesInputs
# from .timestamp_inp import TimestampInputs
from .value_inp import ValueInputs
