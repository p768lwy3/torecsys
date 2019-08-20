from . import _Inputs
from torecsys.utils.decorator import jit_experimental
import numpy as np
import torch
import torch.nn as nn
from typing import List


class FieldAwareMultipleIndexEmbedding(_Inputs):
    r"""FieldAwareSingleIndexEmbedding is a embedding field to pass a list of single index 
    and return a cross field embedding matrix of each vector for a index is :math:`E_{j_{1}, f_{2}}` .
    
    :Reference: 

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_.
    
    """
    @jit_experimental
    def __init__(self, embed_size: int, field_sizes: List[int]):
        r"""initialize field-aware single index embedding field
        
        Args:
            embed_size (int): embedding size
            field_sizes (List[int]): list of field sizes
        """
        super(FieldAwareMultipleIndexEmbedding, self).__init__()
        self.num_fields = len(field_sizes)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_sizes), embed_size) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_sizes)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)
        self.length = embed_size
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Return embedding matrices of inputs
        
        Args:
            inputs (torch.Tensor), shape = (batch size, num fields), dtype = torch.long: torch.Tensor of inputs, \
                where they are the indices of fields (i.e. length of column of torch.Tensor = number of fields) for each row
        
        Returns:
            torch.Tensor, (batch size, num_fields * num_fields, embedding size): embedding matrices :math:`\bm{E} = \bm{\Big[} e_{\text{index}_{i}, \text{feat}_{j}}  \footnotesize{\text{, for} \ i = \text{i-th field} \ \text{and} \ j = \text{j-th field}} \bm{\Big]}`
        """
        inputs = inputs + inputs.new_tensor(self.offsets).unsqueeze(0)
        outputs = torch.cat([self.embeddings[i](inputs) for i in range(self.num_fields)], dim=1)
        return outputs
