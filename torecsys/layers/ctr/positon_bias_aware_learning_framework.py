from typing import Tuple, Dict

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class PositionBiasAwareLearningFrameworkLayer(BaseLayer):
    """
    TODO: missing documentation of this layer.
    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'feature_tensors': ('B', 'E',),
            'session_position': ('B',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', 'E',)
        }

    def __init__(self,
                 input_size: int,
                 max_num_position: int):
        """
        TODO: missing documentation of this layer.

        Args:
            input_size (int):
            max_num_position (int):
        """
        super().__init__()

        self.position_bias = nn.Embedding(max_num_position, input_size)

    def forward(self, position_embed_tensor: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward calculation of PositionBiasAwareLearningFrameworkLayer
        
        Args: position_embed_tensor ((T, T)), shape = ((B, E), (B, )), data_type = (torch.float, torch.long):
            embedded feature tensors of session and position of session in sequence.
        
        Returns:
            T, shape = (B, E), data_type = torch.float: output of PositionBiasAwareLearningFrameworkLayer
        """
        # Get position bias from embedding layer
        # embedder: position_embed_tensor[1], shape = (B, )
        # output: position_embed, shape = (B, E)
        pos = position_embed_tensor[1].rename(None)
        position_embed_bias = self.position_bias(pos)

        # Add position bias to input
        # embedder: position_embed_tensor[0], shape = (B, E)
        # embedder: position_embed_bias, shape = (B, E)
        # output: output, shape = (B, E)
        return position_embed_tensor[0] + position_embed_bias
