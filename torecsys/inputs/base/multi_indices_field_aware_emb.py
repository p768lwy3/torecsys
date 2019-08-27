from . import _Inputs
from torecsys.utils.decorator import jit_experimental
import numpy as np
import torch
import torch.nn as nn
from typing import List


class MultiIndicesFieldAwareEmbedding(_Inputs):
    r"""Base Inputs class for field-aware embedding of multi-indices, which is used in Field Aware 
    Factorization (FFM) or its variants. The shape of output is :math:`(B, N * N, E)`, where the 
    embedding tensor :math"`E_{feat_{i, k}, field_{j}}` are looked up the k-th row from the j-th 
    matrix of i-th feature.
    
    :Reference: 

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_.
    
    """
    @jit_experimental
    def __init__(self, embed_size: int, field_sizes: List[int]):
        r"""Initialize MultiIndicesFieldAwareEmbedding.
        
        Args:
            embed_size (int): Size of embedding tensor
            field_sizes (List[int]): List of inputs fields' sizes
        
        Attributes:
            length (int): Size of embedding tensor.
            num_fields (int): Number of inputs' fields.
            embeddings (torch.nn.ModuleList): ModuleList of embedding modules.
            offsets (T): Tensor of offsets to adjust values of inputs to fit the indices of 
                embedding tensors.
        """
        # refer to parent class
        super(MultiIndicesFieldAwareEmbedding, self).__init__()

        # bind num_field to the length of field_sizes
        self.num_fields = len(field_sizes)

        # create ModuleList of nn.Embedding for each field of inputs
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_sizes), embed_size) for _ in range(self.num_fields)
        ])

        # create offsets to re-index inputs by adding them up
        self.offsets = torch.Tensor((0, *np.cumsum(field_sizes)[:-1])).long().unsqueeze(0)

        # initialize nn.Embedding with xavier_uniform_ initializer
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)
        
        # bind length to embed_size
        self.length = embed_size
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of MultiIndicesFieldAwareEmbedding.
        
        Args:
            inputs (T), shape = (B, N), dtype = torch.long: Tensor of indices in inputs fields.
        
        Returns:
            T, (B, N * N, E), dtype = torch.float: Embedded Inputs: :math:`\bm{E} = \bm{\Big[} e_{\text{index}_{i}, \text{feat}_{j}}  
                \footnotesize{\text{, for} \ i = \text{i-th field} \ \text{and} \ j = \text{j-th field}} \bm{\Big]}`.
        """
        # set offset to adjust values of inputs to fit the indices of embedding tensors
        inputs = inputs + self.offsets

        # concatenate embedded inputs into a single tensor for outputing
        outputs = torch.cat([self.embeddings[i](inputs) for i in range(self.num_fields)], dim=1)
        return outputs
