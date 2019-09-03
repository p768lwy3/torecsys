from . import _EmbLoss
from .functional import skip_gram_loss
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGramLoss(_EmbLoss):
    r"""SkipGramLoss is a module to calculate the loss used in SkipGram algorithm 
    :title:`Tomas Mikolov et al, 2013`[1] which isto calculate the loss by the following formula: 
    :math:`loss = - \sum_{c=1}^{C} u_{j_{c}^{*}} + C log ( \sum_{j'=1}^{v} e^{u_{j'}} )` .
    
    :Reference:

    #. `Tomas Mikolov et al, 2013. Efficient Estimation of Word Representations in Vector Space <https://arxiv.org/abs/1301.3781>`_.
    
    """
    def __init__(self):
        r"""Initialize SkipGramLoss
        """
        # refer to parent class
        super(SkipGramLoss, self).__init__()
    
    def forward(self, 
                content_inputs: torch.Tensor, 
                pos_inputs    : torch.Tensor, 
                neg_inputs    : torch.Tensor) -> torch.Tensor:
        r"""Foward calculation of SkipGramLoss
        
        Args:
            content_inputs (T), shape = (B, 1, E), dtype = torch.float: Predicted scores of content or anchor.
            pos_inputs (T), shape = (B, 1, E), dtype = torch.float: Predicted scores of positive samples.
            neg_inputs (T, shape = (B, Nneg, E), dtype = torch.float: Predicted scores of negative samples.
        
        Returns:
            T, shape = (1, ), dtype = torch.float: Output of SkipGramLoss.
        """
        # calculate skip gram loss
        loss = skip_gram_loss(content_inputs, pos_inputs, neg_inputs)
        return loss
