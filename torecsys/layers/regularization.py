import torch
import torch.nn as nn
from torecsys.utils.operations import regularize
from typing import List, Tuple

class Regularizer(nn.Module):
    r"""nn.Module of Regularizer
    """
    def __init__(self, 
                 weight_decay : float = 0.01,
                 norm         : int   = 2):
        r"""initialize the regularizer module
        
        Args:
            weight_decay (float, optional): multiplier of regularized loss. Defaults to 0.01.
            norm (int, optional): order of norm to calculate regularized loss. Defaults to 2.
        """
        super(Regularizer, self).__init__()

        self.weight_decay = weight_decay
        self.norm = norm
    
    def extra_repr(self) -> str:
        r"""Return information in print-statement of layer.
        
        Returns:
            str: Information of print-statement of layer.
        """
        return 'weight_decay={}, norm={}'.format(self.weight_decay, self.norm)
    
    def forward(self, parameters: List[Tuple[str, nn.Parameter]]) -> torch.Tensor:
        r"""feed forward of regularizer 
        
        Args:
            parameters (List[Tuple[str, nn.Parameter]]): list of tuple of names and paramters 
                to calculate the regularized loss
        
        Returns:
            T, shape = (1, ), dtype = torch.float: regularized loss
        """
        # calculate regularized loss 
        reg_loss = regularize(parameters, self.weight_decay, self.norm)
        
        return reg_loss
