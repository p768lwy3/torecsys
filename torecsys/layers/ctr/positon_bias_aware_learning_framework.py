import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Tuple

class PositionBiasAwareLearningFrameworkLayer(nn.Module):
    def __init__(self,
                 input_size       : int,
                 max_num_position : int):
        # refer to parent class
        super(PositionBiasAwareLearningFrameworkLayer, self).__init__()

        # Initialize Embedding layer
        self.position_bias = nn.Embedding(max_num_position, input_size)
    
    def forward(self, position_embed_tensor: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        r"""Forward calculation of PositionBiasAwareLearningFrameworkLayer
        
        Args:
            position_embed_tensor ((T, T)), shape = ((B, E), (B, )), dtype = (torch.float, torch.long): Embedded feature tensors of session and Position of session in sequence.
        
        Returns:
            T, shape = (B, E), dtype = torch.float: Output of PositionBiasAwareLearningFrameworkLayer
        """
        # Get position bias from embedding layer
        # inputs: position_embed_tensor[1], shape = (B, )
        # output: position_embed, shape = (B, E)
        position_embed_bias = self.position_bias(position_embed_tensor[1])
        
        # Add position bias to input
        # inputs: position_embed_tensor[0], shape = (B, E)
        # inputs: position_embed_bias, shape = (B, E)
        # output: output, shape = (B, E)
        output = position_embed_tensor[0] + position_embed_bias

        return output
