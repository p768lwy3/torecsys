"""
torecsys.models.ltr.losses.pairwise_ranking_loss is a sub model of algorithms of pairwise ranking loss
"""

from typing import Union, Optional

import torch
import torch.nn as nn

from torecsys.losses.ltr import RankingLoss
from torecsys.losses.ltr.functional import apply_mask, parse_margin_ranking_loss, parse_soft_margin_loss
from torecsys.losses.ltr.functional import bayesian_personalized_ranking_loss, hinge_loss, adaptive_hinge_loss
from torecsys.utils import get_reduction


class PairwiseRankingLoss(RankingLoss):
    """
    Base Class of pairwise ranking loss
    """

    def __init__(self):
        super(PairwiseRankingLoss, self).__init__()


class BayesianPersonalizedRankingLoss(PairwiseRankingLoss):
    r"""
    pairwise loss calculated bayesian personalized ranking, by the following equation:
        :math:`loss = \sigma (y_{pos} - y_{neg})` .
    
    :Reference:

    #. `Steffen Rendle et al, 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback
    <https://arxiv.org/abs/1205.2618>`_.

    """

    def __init__(self,
                 reduction: Optional[Union[nn.Module, str]] = 'sum'):
        """
        Initialize BayesianPersonalizedRankingLoss
        
        Args:
            reduction Union[nn.Module, str], optional): reduction method to calculate loss. Defaults to torch.sum
        """
        super().__init__()
        self.reduction = get_reduction(reduction)

    def forward(self,
                pos_out: torch.Tensor,
                neg_out: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward calculation of BayesianPersonalizedRankingLoss
        
        Args:
            pos_out (T), shape = (B, 1): predicted values of positive samples
            neg_out (T), shape = (B, N Neg): predicted values of negative samples
            mask (T, optional), shape = (batch size, ), data_type = torch.bool: boolean tensor to mask loss.
                Defaults to None
        
        Returns:
            T: reduced (masked) loss
        """
        loss = bayesian_personalized_ranking_loss(pos_out, neg_out)
        return self.reduction(apply_mask(loss, mask)) if mask is not None else self.reduction(loss)


class HingeLoss(PairwiseRankingLoss):
    r"""
    HingeLoss is a pairwise ranking loss function which calculated loss with the following equation:
        :math:`loss = max ( 0.0, 1.0 + y_{pos} - y_{neg} )` .
    """

    def __init__(self,
                 margin: float = 1.0,
                 reduction: Optional[nn.Module] = torch.sum):
        """
        Initialize HingeLoss
        
        Args:
            margin (float, optional): Margin size of loss. Defaults to 1.0
            reduction (nn.Module, optional): Reduction method to calculate loss. Defaults to torch.sum
        """
        super().__init__()
        self.margin = margin
        self.reduction = get_reduction(reduction)

    def forward(self,
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward calculation of HingeLoss
        
        Args:
            pos_outputs (T), shape = (B, 1): predicted values of positive samples
            neg_outputs (T), shape = (B, N Neg): predicted values of negative samples
            mask (T, optional), shape = (batch size, ), data_type = torch.bool: boolean tensor to mask loss.
                Defaults to None.
        
        Returns:
            T: reduced (masked) loss
        """
        loss = hinge_loss(pos_outputs, neg_outputs, self.margin)
        return self.reduction(apply_mask(loss, mask)) if mask is not None else self.reduction(loss)


class AdaptiveHingeLoss(PairwiseRankingLoss):
    r"""
    AdaptiveHingeLoss is a pairwise ranking loss function which is a variant of hinge loss and calculated the loss
    between positive samples and the negative samples where their scores are highest, and so the equation will be:
    :math:`loss = max ( 0.0, 1.0 + y_{pos} - max ( y_{neg} ) )` .

    :Reference:

    #. `Jason Weston el at, 2011. WSABIE: Scaling Up To Large Vocabulary Image Annotation
    <https://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf>`_.

    """

    def __init__(self,
                 margin: Optional[float] = 1.0,
                 reduction: Optional[nn.Module] = torch.sum):
        """
        Initialize HingeLoss
        
        Args:
            margin (float, optional): margin size of loss. Defaults to 1.0
            reduction (nn.Module, optional): reduction method to calculate loss. Defaults to torch.sum
        """
        super().__init__()

        self.margin = margin
        self.reduction = get_reduction(reduction)

    def forward(self,
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward calculation of AdaptiveHingeLoss
        
        Args:
            pos_outputs (T), shape = (B, 1): predicted values of positive samples
            neg_outputs (T), shape = (B, N Neg): predicted values of negative samples
            mask (T, optional), shape = (batch size, ), data_type = torch.bool: boolean tensor to mask loss.
                Defaults to None.
        
        Returns:
            T: reduced (masked) loss
        """
        loss = adaptive_hinge_loss(pos_outputs, neg_outputs, self.margin)
        return self.reduction(apply_mask(loss, mask)) if mask is not None else self.reduction(loss)


class TripletLoss(PairwiseRankingLoss):
    r"""
    TripletLoss is a pairwise ranking loss which is used in FaceNet at first, and implemented by PyTorch in model\:
    torch.nn.MarginRankingLoss and torch.nn.SoftMarginLoss. This model is an integration of those losses as a
    standardize calling method with other losses implemented in this package. For the calculation, the loss is
    calculated by :math:`\Big[\left\| x_{anchor} - x_{pos} \right\|_{2}^{2} - \left\| x_{anchor} - x_{neg} \right\|_{
    2}^{2} \Big]_{\text{+}}` .
    
    :Reference:

    #. `Florian Schroff et at, 2015. FaceNet: A Unified Embedding for Face Recognition and Clustering
    <https://arxiv.org/abs/1503.03832>`_.

    """

    def __init__(self,
                 margin: Optional[float] = 1.0,
                 reduction: Optional[str] = 'sum'):
        """
        Initialize TripletLoss
        
        Args:
            margin (float, optional): size of margin. Defaults to 1.0
            reduction (str, optional): method of reduction. Defaults to 'sum'
        """
        super().__init__()

        if margin:
            self.parser = parse_margin_ranking_loss
            self.loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)
        else:
            self.parser = parse_soft_margin_loss
            self.loss = nn.SoftMarginLoss(reduction=reduction)

    def forward(self,
                pos_out: torch.Tensor,
                neg_out: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward calculation of TripletLoss
        
        Args:
            pos_out (T), shape = (B, 1): predicted values of positive samples
            neg_out (T), shape = (B, N Neg): predicted values of negative samples
            mask (T, optional), shape = (B,), data_type = torch.bool: boolean tensor to mask loss. Defaults to None
        
        Returns:
            T: reduced (masked) loss
        """
        if mask is not None:
            pos_out = pos_out[mask]
            neg_out = neg_out[mask]

        y = torch.ones_like(neg_out)
        inputs = self.parser(pos_out, neg_out, y)
        loss = self.loss(**inputs)
        return loss
