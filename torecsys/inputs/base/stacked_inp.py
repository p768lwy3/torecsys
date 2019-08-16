from . import _Inputs
# from .audio_inp import AudioInputs
# from .field_aware_single_index_emb import FieldAwareSingleIndexEmbedding
from .image_inp import ImageInputs
# from .image_list_inp import ImageListInputs
from .list_index_emb import ListIndexEmbedding
# from .pretrained_image_inp import PretrainedImageInputs
# from .pretrained_text_inp import PretrainedTextInputs
from .sequence_index_emb import SequenceIndexEmbedding
from .single_index_emb import SingleIndexEmbedding
# from .text_inp import TextInputs
# from .timeseries_inp import TimeseriesInputs
# from .timestamp_inp import TimestampInputs
from .value_inp import ValueInputs

import torch
import torch.nn as nn
from typing import Dict, List, Tuple

__field_type__ = ["image", "list_index", "sequence_index", "single_index", 
                  "value"]

class StackedInputs(_Inputs):
    r"""StackedInputs is a field of stacked inputs to stack multiple inputs into a row of features vectors
    """
    def __init__(self,
                 field_names: List[str],
                 field_types: List[str],
                 field_sizes: List[int],
                 embed_sizes: List[int],
                 **kwargs):
        r"""initialize stacked inputs field
        
        Args:
            field_names (List[str]): list of string of field name
            field_types (List[str]): list of string of field type
            field_sizes (List[int]): list of integer of field size
            embed_sizes (List[int]): list of integer of embedding size
        
        Kwargs:
            [field_name] (dict): key is field_name string and value = kwargs of dictionary for inputs or embedding field
        
        Raises:
            ValueError: when the length of field_names, field_types, field_sizes and embed_sizes are not same
            ValueError: when the field types contain strings which is not in __field_type__
        """
        super(StackedInputs, self).__init__()
        self.embeddings = nn.ModuleDict()
        self.length = sum(embed_sizes)
        
        num_fields = len(field_names)
        if not all(len(lst) == num_fields for lst in [field_types, field_sizes, embed_sizes]):
            raise ValueError("all inputs list must be the same lengths")

        for fname, ftype, fsize, esize in zip(field_names, field_types, field_sizes, embed_sizes):
            fkwargs = kwargs.get(field_name, {})
            if ftype == "image":
                self.embeddings[fname] = ImageInputs(esize, **fkwargs)
            elif ftype == "list_index":
                self.embeddings[fname] = ListIndexEmbedding(esize, fsize, **fkwargs)
            # elif ftype == "pretrained_image":
            #     self.embeddings[fname] = PretrainedImageInputs(esize, **fkwargs)
            elif ftype == "sequence_index":
                self.embeddings[fname] = SequenceIndexEmbedding(esize, fsize, **fkwargs)
            elif ftype == "single_index":
                self.embeddings[fname] = SingleIndexEmbedding(esize, fsize, **fkwargs)
            elif ftype == "value":
                self.embeddings[fname] = ValueInputs(num_fields=embed_size)
            else:
                raise ValueError("field_name %s with field_type %s is not allowed, Only allow: [%s]." % (fname, ftype, ", ".join(__field_type__)))    
    
    def forward(self, inputs: Dict[str, Tuple[torch.Tensor]]) -> torch.Tensor:
        r"""Return field-stacked features vectors
        
        Args:
            inputs (Dict[str, Tuple[torch.Tensor]]): inputs dictionary, where keys are field names, and values are tuple of torch.Tensor
        
        Raises:
            ValueError: when lengths (i.e. 1-st dimension, batch size) of inputs values are not equal
        
        Returns:
            torch.Tensor, shape = (batch size, 1, embedding size): 2nd dimensional stacked features vectors
        """
        # check if all inputs' values are the same length
        batch_size = list(inputs.values())[0][0].size(0)
        if not all(field_input.size(0) == batch_size for field_inputs in inputs.values() for field_input in field_inputs):
            raise ValueError("lengths of inputs values must be the same.")

        # append the embeddings to a list
        outputs = []
        for field_name in self.field_names:
            outputs.append(self.embeddings[field_name](*inputs[field_name]))

        # unsqueeze(1) if output dim = 2, then concatenate with dim = 2 and return
        outputs = [o.unsqueeze(1) if o.dim() == 2 else o for o in outputs]
        return torch.cat(outputs, dim=2)
