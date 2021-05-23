from typing import List, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn

from torecsys.inputs.base import BaseInput


class MultiIndicesFieldAwareEmbedding(BaseInput):
    r"""
    Base Input class for Field-aware embedding of multiple indices, which is used in Field-aware
    Factorization (FFM) or the variants. The shape of output is :math:`(B, N * N, E)`, where the embedding 
    tensor :math:`E_{feat_{i, k}, field_{j}}` are looked up the k-th row from the j-th tensor of i-th feature.

    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction
        <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_
    
    """
    MultiIndicesFieldAwareEmbedding = TypeVar('MultiIndicesFieldAwareEmbedding')

    def __init__(self,
                 embed_size: int,
                 field_sizes: List[int],
                 device: str = 'cpu',
                 flatten: Optional[bool] = False):
        """
        Initialize MultiIndicesFieldAwareEmbedding.
        
        Args:
            embed_size (int): size of embedding tensor
            field_sizes (List[int]): list of inputs fields' sizes
            device (str): device of torch. Default to cpu.
            flatten (bool, optional): whether outputs is reshape to (B, 1, N * N * E) or not before return.
                Defaults to False

        Attributes:
            length (int): size of embedding tensor multiply by number of fields if flatten is True,
                else Size of embedding tensor
            embeddings (torch.nn.Module): embedding layers
            flatten (bool): flag to show outputs will be flatten or not
            offsets (T): tensor of offsets to adjust values of inputs to fit the indices of embedding tensors
        """
        super().__init__()

        self.num_fields = len(field_sizes)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_sizes), embed_size).to(device) for _ in range(self.num_fields)
        ])
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)
        self.embeddings = self.embeddings.to(device)

        self.offsets = torch.Tensor((0, *np.cumsum(field_sizes)[:-1])).long()
        self.offsets.names = ('N',)
        self.offsets = self.offsets.unflatten('N', (('B', 1,), ('N', self.offsets.size('N'),),))
        self.offsets.to(device)

        self.flatten = flatten
        self.length = embed_size

    def cuda(self, device=None) -> MultiIndicesFieldAwareEmbedding:
        """
        Set MultiIndicesEmbedding to GPU

        Returns:
            MultiIndicesEmbedding: self
        """
        super().cuda(device=device)

        self.offsets = self.offsets.cuda(device)

        return self

    def cpu(self) -> MultiIndicesFieldAwareEmbedding:
        """
        Set MultiIndicesEmbedding to CPU

        Returns:
            MultiIndicesEmbedding: self
        """
        super().cpu()

        self.offsets = self.offsets.cpu()

        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""
        Forward calculation of MultiIndicesFieldAwareEmbedding.
        
        Args:
            inputs (T), shape = (B, N), data_type = torch.long: Tensor of indices in inputs fields.
        
        Returns:
            T, shape = (B, 1, N * N * E) | (B, N * N, E), data_type = torch.float: Embedded Input:
                :math:`\bm{E} = \bm{\Big[} e_{\text{index}_{i}, \text{feat}_{j}} \footnotesize{\text{, for} \ i =
                \text{i-th field} \ \text{and} \ j = \text{j-th field}} \bm{\Big]}`.
        """
        inputs = inputs + self.offsets
        inputs = inputs.rename(None)
        outputs = torch.cat([self.embeddings[i](inputs) for i in range(self.num_fields)], dim=1)

        if self.flatten:
            outputs = outputs.flatten(('N', 'E',), 'E').rename(None).unsqueeze(1)

        outputs.names = ('B', 'N', 'E',)
        return outputs
