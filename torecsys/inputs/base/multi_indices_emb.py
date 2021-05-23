from typing import List, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn

from torecsys.inputs.base import BaseInput


class MultiIndicesEmbedding(BaseInput):
    """
    Base Input class for embedding indices in multi fields of inputs, which is more efficient than
    embedding with a number of SingleIndexEmbedding.
    """

    MultiIndicesEmbedding = TypeVar('MultiIndicesEmbedding')

    def __init__(self,
                 embed_size: Optional[int] = None,
                 field_sizes: Optional[List[int]] = None,
                 nn_embedding: Optional[nn.Parameter] = None,
                 device: str = 'cpu',
                 flatten: Optional[bool] = False,
                 **kwargs):
        """
        Initialize MultiIndicesEmbedding.
        
        Args:
            embed_size (int): size of embedding tensor. Defaults to None
            field_sizes (List[int]): list of inputs fields' sizes. Defaults to None
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None
            device (str): device of torch. Defaults to cpu
            flatten (bool, optional): whether outputs is reshape to (B, 1, N * E) or not before return.
                Defaults to False
        
        Attributes:
            length (int): size of embedding tensor multiply by number of fields if flatten is True,
                else Size of embedding tensor
            embedding (torch.nn.Module): embedding layer
            flatten (bool): flag to show outputs will be flatten or not
            offsets (T): tensor of offsets to adjust values of inputs to fit the indices of embedding tensors
        """
        super().__init__()

        if nn_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        elif sum(field_sizes) is not None and embed_size is not None:
            self.embedding = nn.Embedding(sum(field_sizes), embed_size, **kwargs)
        else:
            raise ValueError('missing required arguments')

        self.embedding = self.embedding.to(device)

        self.offsets = torch.Tensor((0, *np.cumsum(field_sizes)[:-1])).long()
        self.offsets.names = ('N',)
        self.offsets = self.offsets.unflatten('N', (('B', 1,), ('N', self.offsets.size('N'),),))
        self.offsets = self.offsets.to(device)

        self.flatten = flatten

        self.field_size = self.embedding.num_embeddings
        self.embed_size = self.embedding.embedding_dim
        self.padding_idx = self.embedding.padding_idx
        self.length = self.embed_size * len(field_sizes) if self.flatten else self.embed_size

    def cuda(self, device=None) -> MultiIndicesEmbedding:
        """
        Set MultiIndicesEmbedding to GPU
        
        Returns:
            MultiIndicesEmbedding: self
        """
        super().cuda(device=device)

        self.offsets = self.offsets.cuda(device)

        return self

    def cpu(self) -> MultiIndicesEmbedding:
        """
        Set MultiIndicesEmbedding to CPU
        
        Returns:
            MultiIndicesEmbedding: self
        """
        super().cpu()

        self.offsets = self.offsets.cpu()

        return self

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of MultiIndicesEmbedding
        
        Args:
            inputs (T), shape = (B, N), data_type = torch.long: tensor of indices in inputs fields
        
        Returns:
            T, shape = (B, 1, N * E) | (B, N, E), data_type = torch.float:
                outputs of MultiIndicesEmbedding
        """
        inputs = inputs + self.offsets
        outputs = self.embedding(inputs.rename(None))
        outputs.names = ('B', 'N', 'E',)

        if self.flatten:
            outputs = outputs.flatten(('N', 'E',), 'E').rename(None).unsqueeze(1)

        outputs.names = ('B', 'N', 'E',)
        return outputs
