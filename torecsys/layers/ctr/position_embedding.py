from typing import Dict, Tuple

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class PositionEmbeddingLayer(BaseLayer):
    """
    Layer class of Position Embedding

    Position Embedding was used in Personalized Re-ranking Model :title:`Changhua Pei et al, 2019`[1], which is to
    add a trainable tensors per position to the session-based embedding features tensor.

    :Reference:

    `Changhua Pei et al, 2019. Personalized Re-ranking for Recommendation <https://arxiv.org/abs/1904.06813>`_.

    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'L', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', 'L', 'E',)
        }

    def __init__(self, max_num_position: int):
        """
        Initialize PositionEmbedding
        
        Args:
            max_num_position (int): maximum number of position in a sequence
        """
        super().__init__()

        self.bias = nn.Parameter(torch.Tensor(1, max_num_position, 1))
        nn.init.normal_(self.bias)

    def forward(self, session_embed_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of PositionEmbedding
        
        Args:
            session_embed_inputs (T), shape = (B, L, E), data_type = torch.float: embedded feature tensors of session
        
        Returns:
            T, shape = (B, L, E), data_type = torch.float: output of PositionEmbedding
        """
        # Add positional bias to session embedding features
        # embedder: session_embed_inputs, shape = (B, L, E)
        # embedder: self.bias, shape = (1, L, 1)
        # output: output, shape = (B, L, E)
        return session_embed_inputs + self.bias
