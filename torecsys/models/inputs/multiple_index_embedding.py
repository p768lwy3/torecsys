from . import _Inputs
import numpy as np
import torch
import torch.nn as nn
from typing import List


class MultipleIndexEmbedding(_Inputs):
    r"""MultipleIndexEmbedding is a embedding field to pass indices of multiple fields 
    and lookup the embedding vectors at the same time, which will be more efficent
    """
    def __init__(self, 
                 embed_size   : int,
                 field_sizes  : List[int],
                 nn_embedding : nn.Parameter = None,
                 **kwargs):
        r"""initialize the multiple index embedding
        
        Args:
            embed_size (int): embedding size
            field_sizes (List[int]): list of fields' size, which will also be the offset during lookup
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None.
        """
        super(MultipleIndexEmbedding, self).__init__()
        if nn_embedding is not None:
            embed_size = nn_embedding.size(1)
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        else:
            self.embedding = nn.Embedding(sum(field_sizes), embed_size, **kwargs)
        self.offsets = np.array((0, *np.cumsum(field_sizes)[:-1]), dtype=np.long)
        self.length = embed_size
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return embedding vectors of inputs
        
        Args:
            inputs (torch.Tensor), shape = (batch size, num fields), dtype = torch.long: [description]
        
        Returns:
            torch.Tensor, shape = (batch size, num fields, embedding size), dtype = torch.float: [description]
        """
        # add offset to it
        inputs = inputs + inputs.new_tensor(self.offsets).unsqueeze(0)
        outputs = self.embedding(inputs)
        return outputs
