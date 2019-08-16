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
        torch.Tensor, shape = (1, ), dtype = torch.float: regularized loss
    """
    loss = 0.0
    for name, param in parametes:
        if "weight" in name:
            loss += torch.norm(param, p=norm)
    
    return loss * weight_decay
