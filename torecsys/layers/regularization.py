from typing import List, Tuple

import torch
import torch.nn as nn

from torecsys.utils.operations import regularize


class Regularizer(nn.Module):
    """
    Module of Regularizer
    """

    def __init__(self,
                 weight_decay: float = 0.01,
                 norm: int = 2):
        """
        Initialize the regularizer model
        
        Args:
            weight_decay (float, optional): multiplier of regularized loss. Defaults to 0.01
            norm (int, optional): order of norm to calculate regularized loss. Defaults to 2
        """
        super().__init__()

        self.weight_decay = weight_decay
        self.norm = norm

    def extra_repr(self) -> str:
        """
        Return information in print-statement of layer
        
        Returns:
            str: Information of print-statement of layer
        """
        return f'weight_decay={self.weight_decay}, norm={self.norm}'

    def forward(self, parameters: List[Tuple[str, nn.Parameter]]) -> torch.Tensor:
        """
        Forward calculation of Regularizer
        
        Args:
            parameters (List[Tuple[str, nn.Parameter]]): list of tuple of names and parameters to calculate
                the regularized loss
        
        Returns:
            T, shape = (1, ), data_type = torch.float: regularized loss
        """
        return regularize(parameters, self.weight_decay, self.norm)
