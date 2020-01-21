r"""torecsys.models.ltr.losses.pairwise_ranking_loss is a sub module of algorithms of pairwise ranking loss
"""

from . import _RankingLoss
from .functional import apply_mask, margin_ranking_loss_parser, soft_margin_loss_parser
from .functional import bayesian_personalized_ranking_loss, hinge_loss, adaptive_hinge_loss
import torch
import torch.nn as nn
from torecsys.utils import get_reduction
from typing import Callable, Union

class _PairwiseRankingLoss(_RankingLoss):
    r"""Base Class of pairwise ranking loss
    """
    def __init__(self):
        super(_PairwiseRankingLoss, self).__init__()

class BayesianPersonalizedRankingLoss(_PairwiseRankingLoss):
    r"""pairwise loss calculated bayesian personalized ranking, by the following equation: 
    :math:`loss = \sigma (y_{pos} - y_{neg})` .
    
    :Reference:

    #. `Steffen Rendle et al, 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback <https://arxiv.org/abs/1205.2618>`_.

    """
    def __init__(self, 
                 reduction: Union[Callable[[torch.Tensor], torch.Tensor], str] = "sum"):
        r"""Initialize BayesianPersonalizedRankingLoss
        
        Args:
            reduction Union[Callable[T, T], str], optional): reduction method to calculate loss.
                Defaults to torch.sum.
        
        Attributes:
            reduction (Union[Callable[T, T], str]): reduction method to calculate loss.
        """
        # Refer to parent class
        super(BayesianPersonalizedRankingLoss, self).__init__()

        # Bind reduction to reduction
        self.reduction = get_reduction(reduction)
    
    def forward(self,
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        r"""Forward calculation of BayesianPersonalizedRankingLoss
        
        Args:
            pos_outputs (T), shape = (B, 1): Predicted values of positive samples.
            neg_outputs (T), shape = (B, num_neg): Predicted values of negative smaples.
            mask (T, optional), shape = (batch size, ), dtype = torch.bool: Boolean tensor to mask loss. 
                Defaults to None.
        
        Returns:
            T: reduced (masked) loss
        """
        # Calculate loss by functional method
        loss = bayesian_personalized_ranking_loss(pos_outputs, neg_outputs)

        # Apply masking and take reduction on loss
        return self.reduction(apply_mask(loss, mask)) if mask is not None \
            else self.reduction(loss)

class HingeLoss(_PairwiseRankingLoss):
    r"""HingeLoss is a pairwise ranking loss function which calculated loss with the following equation: 
    :math:`loss = max ( 0.0, 1.0 + y_{pos} - y_{neg} )` .
    """
    def __init__(self, 
                 margin: float = 1.0,
                 reduction: Callable[[torch.Tensor], torch.Tensor] = torch.sum):
        r"""Initialize HingeLoss
        
        Args:
            margin (float, optional): Margin size of loss. Defaults to 1.0.
            reduction (Callable[T, T], optional): Reduction method to calculate loss.
                Defaults to torch.sum.
        
        Attributes:
            margin (float): Margin size of loss.
            reduction (Callable[T, T]): Reduction method to calculate loss.
        """
        # Refer to parent class
        super(HingeLoss, self).__init__()

        # Bind margin and reduction to margin and reduction
        self.margin = margin
        self.reduction = get_reduction(reduction)

    def forward(self,
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        r"""Forward calculation of HingeLoss
        
        Args:
            pos_outputs (T), shape = (B, 1): Predicted values of positive samples.
            neg_outputs (T), shape = (B, num_neg): Predicted values of negative smaples.
            mask (T, optional), shape = (batch size, ), dtype = torch.bool: Boolean tensor to mask loss. 
                Defaults to None.
        
        Returns:
            T: reduced (masked) loss
        """
        # Calculate loss by functional method
        loss = hinge_loss(pos_outputs, neg_outputs, self.margin)
        
        # Apply masking and take reduction on loss
        return self.reduction(apply_mask(loss, mask)) if mask is not None \
            else self.reduction(loss)

class AdaptiveHingeLoss(_PairwiseRankingLoss):
    r"""AdaptiveHingeLoss is a pairwise ranking loss function which is a variant of hinge loss and
    calculated the loss between positive samples and the negative samples where their scores are 
    highest, and so, the equation will be: :math:`loss = max ( 0.0, 1.0 + y_{pos} - max ( y_{neg} ) )` .

    :Reference:

    #. `Jason Weston el at, 2011. WSABIE: Scaling Up To Large Vocabulary Image Annotation <http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf>`_.

    """
    def __init__(self, 
                 margin: float = 1.0,
                 reduction: Callable[[torch.Tensor], torch.Tensor] = torch.sum):
        r"""Initialize HingeLoss
        
        Args:
            margin (float, optional): Margin size of loss. Defaults to 1.0.
            reduction (Callable[T, T], optional): Reduction method to calculate loss.
                Defaults to torch.sum.
        
        Attributes:
            margin (float): Margin size of loss.
            reduction (Callable[T, T]): Reduction method to calculate loss.
        """
        # Refer to parent class
        super(AdaptiveHingeLoss, self).__init__()

        # Bind margin and reduction to margin and reduction
        self.margin = margin
        self.reduction = get_reduction(reduction)
    
    def forward(self,
                pos_outputs: torch.Tensor,
                neg_outputs: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        r"""Forward calculation of AdaptiveHingeLoss
        
        Args:
            pos_outputs (T), shape = (B, 1): Predicted values of positive samples.
            neg_outputs (T), shape = (B, num_neg): Predicted values of negative smaples.
            mask (T, optional), shape = (batch size, ), dtype = torch.bool: Boolean tensor to mask loss. 
                Defaults to None.
        
        Returns:
            T: reduced (masked) loss
        """
        # Calculate loss by functional method
        loss = adaptive_hinge_loss(pos_outputs, neg_outputs, self.margin)

        # Apply masking and take reduction on loss
        return self.reduction(apply_mask(loss, mask)) if mask is not None \
            else self.reduction(loss)

class TripletLoss(_PairwiseRankingLoss):
    r"""TripletLoss is a pairwise ranking loss which is used in FaceNet at first, 
    and implemented by PyTorch in module\: torch.nn.MarginRankingLoss and torch.nn.SoftMarginLoss. 
    This module is an integration of those losses as a standardize calling method with other losses 
    implemented in this package. For the calculation, the loss is calculated by 
    :math:`\Big[\left\| x_{anchor} - x_{pos} \right\|_{2}^{2} - \left\| x_{anchor} - x_{neg} \right\|_{2}^{2} \Big]_{\text{+}}` .
    
    :Reference:

    #. `Florian Schroff et at, 2015. FaceNet: A Unified Embedding for Face Recognition and Clustering <https://arxiv.org/abs/1503.03832>`_.

    """
    def __init__(self, 
                 margin    : float = 1.0, 
                 reduction : str = None):
        r"""Initialize TripletLoss
        
        Args:
            margin (float, optional): size of margin. Defaults to 1.0.
            reduction (str, optional): method of reduction. Defaults to None.
        """
        # Refer to parent class
        super(TripletLoss, self).__init__()

        # Initialize module with input margin
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
        r"""Forward calculation of TripletLoss
        
        Args:
            pos_outputs (T), shape = (B, 1): Predicted values of positive samples.
            neg_outputs (T), shape = (B, num_neg): Predicted values of negative smaples.
            mask (T, optional), shape = (batch size, ), dtype = torch.bool: Boolean tensor to mask loss. 
                Defaults to None.
        
        Returns:
            T: reduced (masked) loss
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
