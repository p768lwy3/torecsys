r"""torecsys.losses is a sub module of implementation of losses in recommendation system.
"""
import torch.nn.modules.loss

class _Loss(torch.nn.modules.loss._Loss):
    r"""General Loss class.
    """
    def __init__(self, 
                 size_average : bool = None,
                 reduce       : bool = None,
                 reduction    : str  = "mean"):
        r"""Initialize _Loss
        
        Args:
            size_average (bool, optional): [description]. Defaults to None.
            reduce (bool, optional): [description]. Defaults to None.
            reduction (str, optional): [description]. Defaults to "mean".
        
        Attributes:
            reduction (str): A string of reduction method to calculate loss.
                Allows: ["mean"|"sum"|None]
        """
        # refer to parent class
        super(_Loss, self).__init__(size_average, reduce, reduction)

from .ctr import *
from .emb import *
from .ltr import *
