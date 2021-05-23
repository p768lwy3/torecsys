"""
torecsys.models.ltr.losses.groupwise_ranking_loss is a sub model of algorithms of groupwise ranking loss
"""

import torch

from torecsys.losses.ltr import RankingLoss
from torecsys.losses.ltr.functional import apply_mask, listnet_loss


class GroupwiseRankingLoss(RankingLoss):
    """
    Base Class of groupwise ranking loss
    """

    def __init__(self):
        super().__init__()


class ListnetLoss(GroupwiseRankingLoss):
    r"""
    listnet groupwise ranking loss is a variant of cross-entropy to do ranking with a list of inputs,
        by the following formula: :math:`loss = \sum \text{Softmax} (y_{true}) * \text{log} (\text{Softmax}
        (\^{y}))` .

    :Reference:

    #. `Zhe Cao et al, 2007. Learning to Rank: From Pairwise Approach to Listwise Approach
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf>`_.

    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(y_hat: torch.Tensor,
                y_true: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward calculation of listnet groupwise ranking loss
        
        Args:
            y_hat (T), shape = (B, L,), data_type = torch.float: predicted ranking scores
            y_true (T), shape = (B, L,), data_type = torch.float: true ranking scores,
                e.g. [Excellent(4), Perfect(3), Good(2), Fair(1), Bad(0)]
            mask (T, optional), shape = (B,), data_type = torch.bool: boolean tensor to mask training loss.
                defaults to None
        
        Returns:
            torch.Tensor: aggregated (masked) loss
        """
        if mask is not None:
            y_hat = apply_mask(y_hat, mask)
            y_true = apply_mask(y_true, mask)

        loss = listnet_loss(y_hat, y_true)
        return loss
