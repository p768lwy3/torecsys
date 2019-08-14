r"""torecsys.models.ltr.losses.pairwise_ranking_loss is a sub module of algorithms of pairwise ranking loss
"""

from . import _RankingLoss
from .functional import apply_mask, margin_ranking_loss_parser, soft_margin_loss_parser
from .functional import bayesian_personalized_ranking_loss, hinge_loss, adaptive_hinge_loss
import torch
import torch.nn as nn


class _PairwiseRankingLoss(_RankingLoss):
    r"""Base Class of pairwise ranking loss
    """
    def __init__(self):
        super(_PairwiseRankingLoss, self).__init__()


class BayesianPersonalizedRankingLoss(_PairwiseRankingLoss):
    r"""pairwise loss calculated bayesian personalized ranking, by the following equation: 
    :math:`loss = \sigma (y_{pos} - y_{neg})` .
    
    :Reference:

    #. `Steffen Rendle et al, 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback <https://arxiv.org/abs/1205.2618>`

    """
    def __init__(self):
        super(BayesianPersonalizedRankingLoss, self).__init__()
    
    def forward(self,
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        r"""feed forward of pairwise ranking loss with bayesian personalized ranking
        
        Args:
            pos_outputs (torch.Tensor), shape = (batch size, 1): scores of positive items
            neg_outputs (torch.Tensor), shape = (batch size, number of negative samples): scores of sampled negative items
            mask (torch.Tensor, optional), shape = (batch size, ), dtype = torch.bool: boolean tensor to mask training loss. Defaults to None.
        
        Returns:
            torch.Tensor: aggregated (masked) loss
        """
        loss = bayesian_personalized_ranking_loss(pos_outputs, neg_outputs)
        return apply_mask(loss, mask) if mask is not None else loss.mean()


class HingeLoss(_PairwiseRankingLoss):
    r"""HingeLoss is a pairwise ranking loss function which calculated loss with the following equation: 
    :math:`loss = max ( 0.0, 1.0 + y_{pos} - y_{neg} )` .
    """
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self,
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        r"""feed forward of pairwise hinge ranking loss

        Args:
            pos_outputs (torch.Tensor), shape = (batch size, 1): scores of positive items
            neg_outputs (torch.Tensor), shape = (batch size, number of negative samples): scores of sampled negative items
            mask (torch.Tensor, optional), shape = (batch size, ), dtype = torch.bool: boolean tensor to mask training loss. Defaults to None.
        
        Returns:
            torch.Tensor: aggregated (masked) loss
        """
        loss = hinge_loss(pos_outputs, neg_outputs)
        return apply_mask(loss, mask) if mask is not None else loss.mean()


class AdaptiveHingeLoss(_PairwiseRankingLoss):
    r"""AdaptiveHingeLoss is a pairwise ranking loss function which is a variant of hinge loss and
    calculated the loss between positive samples and the negative samples where their scores are 
    highest, and so, the equation will be: :math:`loss = max ( 0.0, 1.0 + y_{pos} - max ( y_{neg} ) )` .

    :Reference:

    #. `Jason Weston el at, 2011. WSABIE: Scaling Up To Large Vocabulary Image Annotation <http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf>`

    """
    def __init__(self):
        super(AdaptiveHingeLoss, self).__init__()
    
    def forward(self,
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        r"""feed forward of pairwise adaptive hinge ranking loss

        Args:
            pos_outputs (torch.Tensor), shape = (batch size, 1): scores of positive items
            neg_outputs (torch.Tensor), shape = (batch size, number of negative samples): scores of sampled negative items
            mask (torch.Tensor, optional), shape = (batch size, ), dtype = torch.bool: boolean tensor to mask training loss. Defaults to None.
        
        Returns:
            torch.Tensor: aggregated (masked) loss
        """
        loss = adaptive_hinge_loss(pos_outputs, neg_outputs)
        return apply_mask(loss, mask) if mask is not None else loss.mean()


class TripletLoss(_PairwiseRankingLoss):
    r"""TripletLoss is a pairwise ranking loss which is used in FaceNet at first, 
    and implemented by PyTorch in module\: torch.nn.MarginRankingLoss and torch.nn.SoftMarginLoss. 
    This module is an integration of those losses as a standardize calling method with other losses 
    implemented in this package. For the calculation, the loss is calculated by 
    :math:`\Big[\left\| x_{anchor} - x_{pos} \right\|_{2}^{2} - \left\| x_{anchor} - x_{neg} \right\|_{2}^{2} \Big]_{\text{+}}` .
    
    :Reference:

    #. `Florian Schroff et at, 2015. FaceNet: A Unified Embedding for Face Recognition and Clustering <https://arxiv.org/abs/1503.03832>`

    """
    def __init__(self, 
                 margin    : float = 1.0, 
                 reduction : str = None):
        r"""initialize triplet loss module
        
        Args:
            margin (float, optional): size of margin. Defaults to 1.0.
            reduction (str, optional): method of reduction. Defaults to None.
        """
        super(TripletLoss, self).__init__()
        if margin:
            self.parser = margin_ranking_loss_parser
            self.loss = nn.MarginRankingLoss(margin=margin, reduction=reduction)
        else:
            self.parser = soft_margin_loss_parser
            self.loss = nn.SoftMarginLoss(reduction=reduction)
    
    def forward(self,
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        r"""feed forward of pairwise triplet ranking loss

        Args:
            pos_outputs (torch.Tensor), shape = (batch size, 1): scores of positive items
            neg_outputs (torch.Tensor), shape = (batch size, number of negative samples): scores of sampled negative items
            mask (torch.Tensor, optional), shape = (batch size, ), dtype = torch.bool: boolean tensor to mask training loss. Defaults to None.
        
        Returns:
            torch.Tensor: aggregated (masked) loss
        """
        # masking inputs if needed
        if mask is not None:
            pos_outputs = apply_mask(pos_outputs, mask)
            neg_outputs = apply_mask(neg_outputs, mask)
        
        # create the target ones_liken tensor - y
        y = torch.ones_like(neg_outputs)
        
        # create inputs by parsing with self.parser
        inputs = self.parser(pos_outputs, neg_outputs, y)

        # calculate the loss
        loss = self.loss(**inputs)

        return loss
