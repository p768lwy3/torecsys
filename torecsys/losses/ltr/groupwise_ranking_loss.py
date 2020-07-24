r"""torecsys.models.ltr.losses.groupwise_ranking_loss is a sub module of algorithms of 
groupwise ranking loss
"""

import torch

from . import _RankingLoss
from .functional import apply_mask, listnet_loss


class _GroupwiseRankingLoss(_RankingLoss):
    r"""Base Class of groupwise ranking loss
    """

    def __init__(self):
        super(_GroupwiseRankingLoss, self).__init__()


class ListnetLoss(_GroupwiseRankingLoss):
    r"""listnet groupwise ranking loss is a variant of cross-entropy to do ranking with a list
    of inputs, by the following formula:
    :math:`loss = \sum \text{Softmax} (y_{true}) * \text{log} (\text{Softmax} (\^{y}))` .

    :Reference:

    #. `Zhe Cao et al, 2007. Learning to Rank: From Pairwise Approach to Listwise Approach
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf>`_.

    """

    def __init__(self):
        super(ListnetLoss, self).__init__()

    def forward(self,
                y_hat: torch.Tensor,
                y_true: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        r"""feed forward of listnet groupwise ranking loss
        
        Args: y_hat (torch.Tensor), shape = (batch size, sequence len), dtype = torch.float: Predicted Ranking scores
        y_true (torch.Tensor), shape = (batch size, sequence len ), dtype = torch.float: True Ranking scores,
        e.g. [Excellent(4), Perfect(3), Good(2), Fair(1), Bad(0)] mask (torch.Tensor, optional), shape = (batch size,
        ), dtype = torch.bool: boolean tensor to mask training loss. Defaults to None.
        
        Returns:
            torch.Tensor: aggregated (masked) loss
        """
        # masking inputs if needed
        if mask is not None:
            y_hat = apply_mask(y_hat, mask)
            y_true = apply_mask(y_true, mask)

        # calculate the loss
        loss = listnet_loss(y_hat, y_true)
        return loss
