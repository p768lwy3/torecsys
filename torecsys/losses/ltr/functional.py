r"""torecsys.losses.ltr.functional is a sub module of functions for the implementation of losses in 
learning-to-ranking.
"""

import torch
from typing import Dict


# loss masking
def apply_mask(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Masking training loss by a boolean tensor to drop parts of training loss.
    
    Args:
        loss (T), shape = (B, None), dtype = torch.float: Training loss.
        mask (T), shape = (B, ), dtype = torch.bool: Boolean tensor for masking.
    
    Returns:
        T, shape = (1, ), dtype = torch.float: Aggregated masked loss.
    """
    loss = loss[mask]
    return loss.sum() / mask.sum()


# pointwise ranking loss
def pointwise_logistic_ranking_loss(pout: torch.Tensor, nout: torch.Tensor) -> torch.Tensor:
    r"""Calculation of Pairwise Logistic Ranking Loss.
    
    Args:
        pout (T), shape = (B, 1), dtype = torch.float: Predicted scores of positive samples.
        nout (T), shape = (B, Nneg), dtype = torch.float: Predicted scores of negative samples.
    
    Returns:
        T, shape = (B, Nneg), dtype = torch.float: Output of pointwise_logistic_ranking_loss.
    """
    loss = (1.0 - torch.sigmoid(pout)) + torch.sigmoid(nout)
    return loss


# pairwise ranking loss
def bayesian_personalized_ranking_loss(pout: torch.Tensor, nout: torch.Tensor) -> torch.Tensor:
    r"""Calculation of Bayesian Personalized Ranking Loss.
    
    Args:
        pout (T), shape = (B, 1), dtype = torch.float: Predicted scores of positive samples.
        nout (T), shape = (B, Nneg), dtype = torch.float: Predicted scores of negative samples.
    
    Returns:
        T, shape = (B, Nneg), dtype = torch.float: Output of bayesian_personalized_ranking_loss.
    
    :Reference:

    #. `Steffen Rendle et al, 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback <https://arxiv.org/abs/1205.2618>`_.

    """
    loss = - (pout - nout).sigmoid().log()
    return loss

def hinge_loss(pout: torch.Tensor, nout: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    r"""Calculation of Hinge Loss.
    
    Args:
        pout (T), shape = (B, 1): Predicted scores of positive samples.
        nout (T), shape = (B, Nneg): Predicted scores of negative samples.
        margin (float): margin parameter in hinge loss.
    
    Returns:
        T, shape = (B, Nneg), dtype = torch.float: Output of hinge_loss.
    """
    loss = torch.clamp(margin - pout + nout, min=0.0)
    return loss

def adaptive_hinge_loss(pout: torch.Tensor, nout: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    r"""Calculation of Adaptive Hinge Loss.
    
    Args:
        pout (T), shape = (B, 1): Predicted scores of positive samples.
        nout (T), shape = (B, Nneg): Predicted scores of negative samples.
        margin (float): margin parameter in hinge loss.
    
    Returns:
        T, shape = (B, 1), dtype = torch.float: Output of adaptive_hinge_loss.
    
    :Reference:

    #. `Jason Weston el at, 2011. WSABIE: Scaling Up To Large Vocabulary Image Annotation <http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf>`_.

    """
    highest_nout, _ = torch.max(nout, 0)
    return hinge_loss(pout, highest_nout.unsqueeze(-1), margin)


# parser of ranking loss
def margin_ranking_loss_parser(pout  : torch.Tensor, 
                               nout  : torch.Tensor, 
                               target: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""Parse positive outputs, negative outputs and target to a dictionary as a kwargs inputs
    for nn.MarginRankingLoss
    
    Args:
        pout (T), shape = (B, 1): Predicted scores of positive samples.
        nout (T)), shape = (B, Nneg): Predicted scores of negative samples.
        target (T), shape = (B, Nneg): Tensors of ones created by torch.ones_like(nout).
    
    Returns:
        Dict[str, T]: Dictionary to be passed to nn.MarginRankingLoss.
    
    Key-Values:
        input1: pout, shape = (B, 1)
        input2: nout, shape = (B, Nneg)
        target: target, shape = (B, Nneg)
    """
    return {"input1": pout, "input2": nout, "target": target}

def soft_margin_loss_parser(pout  : torch.Tensor, 
                            nout  : torch.Tensor, 
                            target: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""Parse positive outputs, negative outputs and target to a dictionary as a kwargs inputs
    for nn.SoftMarginLoss
    
    Args:
        pout (T), shape = (B, 1): Predicted scores of positive samples.
        nout (T)), shape = (B, Nneg): Predicted scores of negative samples.
        target (T), shape = (B, Nneg): Tensors of ones created by torch.ones_like(nout).
    
    Returns:
        Dict[str, T]: Dictionary to be passed to nn.SoftMarginLoss.
    
    Key-Values:
        input: (pout - nout), shape = (B, Nneg)
        target: target, shape = (B, Nneg)
    """
    return {"input": pout - nout, "target": target}


# groupwise ranking loss
def listnet_loss(yhat: torch.Tensor, ytrue: torch.Tensor) -> torch.Tensor:
    r"""Cross-Entropy implementation in ListNet by the following formula:
    :math:`loss = \sum \text{Softmax} (y_{true}) * \text{log} (\text{Softmax} (\^{y}))` .
    
    Args:
        yhat (T), shape = (B, sequence len), dtype = torch.float: Predicted Ranking scores
        ytrue (T), shape = (B, sequence len ), dtype = torch.float: True Ranking scores, e.g. [Excellent(4), Perfect(3), Good(2), Fair(1), Bad(0)] 
    
    Returns:
        T: cross-entropy loss 
    
    :Reference:

    #. `Zhe Cao et al, 2007. Learning to Rank: From Pairwise Approach to Listwise Approach <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf>`_.
    
    """
    # calculate softmax for each row of sample
    phat = torch.softmax(yhat, dim=0)
    ptrue = torch.softmax(ytrue, dim=0)

    # calculate loss
    loss = - (ptrue * phat.log()).sum()

    return loss
