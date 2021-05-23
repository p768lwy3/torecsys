from typing import Optional

import torch
import torch.nn as nn

from torecsys.inputs.base import BaseInput


class SingleIndexEmbedding(BaseInput):
    """
    Base Input class for embedding a single index of a input field
    """

    def __init__(self,
                 embed_size: int,
                 field_size: int,
                 padding_idx: Optional[int] = None,
                 nn_embedding: Optional[nn.Parameter] = None,
                 **kwargs):
        """
        Initialize SingleIndexEmbedding
        
        Args:
            embed_size (int): size of embedding tensor
            field_size (int): size of inputs field
            padding_idx (int, optional): padding index. Defaults to None
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None
        
        Arguments:
            length (int): size of embedding tensor
            embedding (torch.nn.Module): embedding layer
            schema (namedtuple): list of string of field names to be embedded
        """
        super().__init__()

        if nn_embedding is not None:
            embed_size = nn_embedding.size('E')
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        else:
            embed_size = embed_size
            self.embedding = nn.Embedding(field_size, embed_size, padding_idx=padding_idx, **kwargs)

        self.length = embed_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of SingleIndexEmbedding
        
        Args:
            inputs (T), shape = (B, 1), data_type = torch.long: tensor of indices in inputs fields
        
        Returns:
            T, (B, 1, E): outputs of SingleIndexEmbedding
        """
        # get embedding tensor from embedding model
        inputs = inputs.long()
        embedded_tensor = self.embedding(inputs.rename(None))
        embedded_tensor.names = ('B', 'N', 'E',)
        return embedded_tensor
