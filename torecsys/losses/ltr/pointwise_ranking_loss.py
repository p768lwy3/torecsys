"""
torecsys.models.ltr.losses.pointwise_ranking_loss is a sub model of algorithms of pointwise ranking loss
"""
from typing import Optional

import torch

from torecsys.losses.ltr import RankingLoss
from torecsys.losses.ltr.functional import apply_mask, pointwise_logistic_ranking_loss


class PointwiseRankingLoss(RankingLoss):
    """
    Base Class of pointwise ranking loss
    """

    def __init__(self):
        super().__init__()


class PointwiseLogisticLoss(PointwiseRankingLoss):
    """
    Pointwise logistic loss
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pos_out: torch.Tensor,
                neg_out: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""
        Forward calculation pointwise logistic ranking loss by calculating
            :math:`\text{loss} = (1.0 - \sigma (y_{pos})) + \sigma (y_{neg})`
        
        Args:
            pos_out (T), shape = (B, 1,): scores of positive items
            neg_out (T), shape = (B, N Neg,): scores of sampled negative items
            mask (T, optional), shape = (B,), data_type = torch.bool: boolean tensor to mask training loss.
                Defaults to None
        
        Returns:
            torch.Tensor: aggregated (masked) loss
        """
        loss = pointwise_logistic_ranking_loss(pos_out, neg_out)
        return apply_mask(loss, mask) if mask is not None else loss.mean()
