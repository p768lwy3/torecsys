"""
torecsys.losses.ltr.functional is a sub model of functions for the implementation of losses in learning-to-ranking.
"""

from typing import Dict

import torch


def apply_mask(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Masking training loss by a boolean tensor to drop parts of training loss.
    
    Args:
        loss (T), shape = (B, None), data_type = torch.float: training loss
        mask (T), shape = (B, ), data_type = torch.bool: boolean tensor for masking
    
    Returns:
        T, shape = (1, ), data_type = torch.float: aggregated masked loss.
    """
    loss = loss[mask]
    return loss.sum() / mask.sum()


def pointwise_logistic_ranking_loss(p_out: torch.Tensor, n_out: torch.Tensor) -> torch.Tensor:
    """
    Calculation of Pairwise Logistic Ranking Loss.
    
    Args:
        p_out (T), shape = (B, 1), data_type = torch.float: predicted scores of positive samples
        n_out (T), shape = (B, NNeg), data_type = torch.float: predicted scores of negative samples
    
    Returns:
        T, shape = (B, NNeg), data_type = torch.float: output of pointwise_logistic_ranking_loss.
    """
    loss = (1.0 - torch.sigmoid(p_out)) + torch.sigmoid(n_out)
    return loss


def bayesian_personalized_ranking_loss(p_out: torch.Tensor, n_out: torch.Tensor) -> torch.Tensor:
    """
    Calculation of Bayesian Personalized Ranking Loss.
    
    Args:
        p_out (T), shape = (B, 1), data_type = torch.float: predicted scores of positive samples
        n_out (T), shape = (B, NNeg), data_type = torch.float: predicted scores of negative samples
    
    Returns:
        T, shape = (B, NNeg), data_type = torch.float: output of bayesian_personalized_ranking_loss
    
    :Reference:

    #. `Steffen Rendle et al, 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback
    <https://arxiv.org/abs/1205.2618>`_.

    """
    loss = - (p_out - n_out).sigmoid().log()
    return loss


def hinge_loss(p_out: torch.Tensor, n_out: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Calculation of Hinge Loss.
    
    Args:
        p_out (T), shape = (B, 1): predicted scores of positive samples
        n_out (T), shape = (B, N Neg): predicted scores of negative samples
        margin (float): margin parameter in hinge loss
    
    Returns:
        T, shape = (B, NNeg), data_type = torch.float: output of hinge_loss
    """
    n_out_size = list(n_out.size())
    p_out_t = p_out.repeat(1, n_out_size[1])
    margin_t = torch.Tensor([margin]).repeat(n_out_size[0], n_out_size[1])
    loss = torch.clamp(margin_t - p_out_t + n_out, min=0.0)
    return loss


def adaptive_hinge_loss(p_out: torch.Tensor, n_out: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    """
    Calculation of Adaptive Hinge Loss.
    
    Args:
        p_out (T), shape = (B, 1): predicted scores of positive samples
        n_out (T), shape = (B, N Neg): predicted scores of negative samples
        margin (float): margin parameter in hinge loss
    
    Returns:
        T, shape = (B, 1), data_type = torch.float: output of adaptive_hinge_loss.
    
    :Reference:

    #. `Jason Weston el at, 2011. WSABIE: Scaling Up To Large Vocabulary Image Annotation
    <https://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf>`_.

    """
    highest_n_out, _ = torch.max(n_out, dim=1, keepdim=True)
    return hinge_loss(p_out, highest_n_out.unsqueeze(-1), margin)


def parse_margin_ranking_loss(p_out: torch.Tensor,
                              n_out: torch.Tensor,
                              target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Parse positive outputs, negative outputs and target to a dictionary as a kwargs embedder for nn.MarginRankingLoss
    
    Args:
        p_out (T), shape = (B, 1): predicted scores of positive samples
        n_out (T)), shape = (B, N Neg): predicted scores of negative samples
        target (T), shape = (B, N Neg): tensors of ones created by torch.ones_like(n_out)
    
    Returns:
        Dict[str, T]: dictionary to be passed to nn.MarginRankingLoss
    
    Key-Values:
        input1: p_out, shape = (B, 1)
        input2: n_out, shape = (B, N Neg)
        target: target, shape = (B, N Neg)
    """
    return {'input1': p_out, 'input2': n_out, 'target': target}


def parse_soft_margin_loss(p_out: torch.Tensor,
                           n_out: torch.Tensor,
                           target: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Parse positive outputs, negative outputs and target to a dictionary as a kwargs embedder for nn.SoftMarginLoss
    
    Args:
        p_out (T), shape = (B, 1): predicted scores of positive samples
        n_out (T)), shape = (B, N Neg): predicted scores of negative samples
        target (T), shape = (B, N Neg): tensors of ones created by torch.ones_like(n_out)
    
    Returns:
        Dict[str, T]: dictionary to be passed to nn.SoftMarginLoss
    
    Key-Values:
        input: (p_out - n_out), shape = (B, N Neg)
        target: target, shape = (B, N Neg)
    """
    return {'input': p_out - n_out, 'target': target}


def listnet_loss(y_hat: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    r"""Cross-Entropy implementation in ListNet by the following formula:
    :math:`loss = \sum \text{Softmax} (y_{true}) * \text{log} (\text{Softmax} (\^{y}))`
    
    Args:
        y_hat (T), shape = (B, sequence len), data_type = torch.float: predicted Ranking scores
        y_true (T), shape = (B, sequence len ), data_type = torch.float: true Ranking scores,
            e.g. [Excellent(4), Perfect(3), Good(2), Fair(1), Bad(0)]
    
    Returns:
        T: cross-entropy loss 
    
    :Reference:

    #. `Zhe Cao et al, 2007. Learning to Rank: From Pairwise Approach to Listwise Approach
    <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf>`_.
    
    """
    # calculate softmax for each row of sample
    p_hat = torch.softmax(y_hat, dim=0)
    p_true = torch.softmax(y_true, dim=0)

    # calculate loss
    loss = - (p_true * p_hat.log()).sum()

    return loss
