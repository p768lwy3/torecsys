r"""torecsys.models.ltr.losses.pointwise_ranking_loss is a sub module of algorithms of 
pointwise ranking loss
"""

from . import _RankingLoss
from .functional import apply_mask, pointwise_logistic_ranking_loss


class _PointwiseRankingLoss(_RankingLoss):
    r"""Base Class of pointwise ranking loss
    """
    def __init__(self):
        super(_PointwiseRankingLoss, self).__init__()


class PointwiseLogisticLoss(_PointwiseRankingLoss):
    r"""pointwise logitic loss
    """
    def __init__(self):
        super(PointwiseLogisticLoss, self).__init__()

    def forward(self, 
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        r"""feed forward of pointwise logistic ranking loss by calculating
        :math:`\text{loss} = (1.0 - \sigma (y_{pos})) + \sigma (y_{neg})`
        
        Args:
            pos_outputs (torch.Tensor), shape = (batch size, 1): scores of positive items
            neg_outputs (torch.Tensor), shape = (batch size, number of negative samples): scores of sampled negative items
            mask (torch.Tensor, optional), shape = (batch size, ), dtype = torch.bool: boolean tensor to mask training loss. Defaults to None.
        
        Returns:
            torch.Tensor: aggregated (masked) loss
        """
        loss = pointwise_logistic_ranking_loss(pos_outputs, neg_outputs)
        return apply_mask(loss, mask) if mask is not None else loss.mean()
