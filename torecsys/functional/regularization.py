import torch
import torch.nn as nn
from texttable import Texttable
from typing import List, Tuple


def regularize(parametes    : List[Tuple[str, nn.Parameter]], 
               weight_decay : float = 0.01,
               norm         : int = 2) -> torch.Tensor:
    r"""function to calculate p-th order regularization of paramters in the model
    
    Args:
        parametes (List[Tuple[str, nn.Parameter]]): list of tuple of names and paramters to calculate the regularized loss
        weight_decay (float, optional): multiplier of regularized loss. Defaults to 0.01.
        norm (int, optional): order of norm to calculate regularized loss. Defaults to 2.
    
    Returns:
        T, shape = (1, ), dtype = torch.float: regularized loss
    """
    loss = 0.0
    for name, param in parametes:
        if "weight" in name:
            loss += torch.norm(param, p=norm)
    
    return loss * weight_decay


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
    
    def forward(self, parameters: List[Tuple[str, nn.Parameter]]) -> torch.Tensor:
        r"""feed forward of regularizer 
        
        Args:
            parameters (List[Tuple[str, nn.Parameter]]): list of tuple of names and paramters to calculate the regularized loss
        
        Returns:
            T, shape = (1, ), dtype = torch.float: regularized loss
        """
        # calculate regularized loss 
        reg_loss = regularize(parameters, self.weight_decay, self.norm)
        
        return reg_loss
