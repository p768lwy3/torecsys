import torch
from typing import Dict


def apply_mask(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""masking training loss with a boolean tensor, which is to remove parts of samples' training loss
    
    Args:
        loss (torch.Tensor), shape = (batch size, None), dtype = torch.float: training loss of model
        mask (torch.Tensor), shape = (batch size, ), dtype = torch.bool: boolean tensor to mask training loss
    
    Returns:
        torch.Tensor: aggregated masked loss
    """
    loss = loss[masking]
    return loss.sum() / mask.sum()


# pointwise ranking loss
def pointwise_logistic_ranking_loss(pout: torch.Tensor, nout: torch.Tensor) -> torch.Tensor:
    r"""logistic loss function
    
    Args:
        pout (torch.Tensor), shape = (batch size, 1): scores of positive items
        nout (torch.Tensor), shape = (batch size, number of negative samples): scores of sampled negative items
    
    Returns:
        torch.Tensor, shape = (batch size, number of negative samples): loss for each negative samples
    """
    loss = (1.0 - torch.sigmoid(pout)) + torch.sigmoid(nout)
    return loss


# pairwise ranking loss
def bayesian_personalized_ranking_loss(pout: torch.Tensor, nout: torch.Tensor) -> torch.Tensor:
    r"""bayesian personalized ranking loss function
    
    Args:
        pout (torch.Tensor), shape = (batch size, 1): scores of positive items
        nout (torch.Tensor), shape = (batch size, number of negative samples): scores of sampled negative items
    
    Returns:
        torch.Tensor, shape = (batch size, number of negative samples): loss for each negative samples
    
    :Reference:
    #. `Steffen Rendle et al, 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback <https://arxiv.org/abs/1205.2618>`
    """
    loss = (1.0 - torch.sigmoid(pout - nout))
    return loss

def hinge_loss(pout: torch.Tensor, nout: torch.Tensor) -> torch.Tensor:
    r"""hinge pairwise loss function
    
    Args:
        pout (torch.Tensor), shape = (batch size, 1): scores of positive items
        nout (torch.Tensor), shape = (batch size, number of negative samples): scores of sampled negative items
    
    Returns:
        torch.Tensor, shape = (batch size, number of negative samples): loss for each negative samples
    """
    loss = torch.clamp(1.0 + pout - nout, 0.0)
    return loss

def adaptive_hinge_loss(pout: torch.Tensor, nout: torch.Tensor) -> torch.Tensor:
    r"""adaptive hinge pairwise loss function
    
    Args:
        pout (torch.Tensor), shape = (batch size, 1): scores of positive items
        nout (torch.Tensor), shape = (batch size, number of negative samples): scores of sampled negative items
    
    Returns:
        torch.Tensor, shape = (batch size, 1): loss for each negative samples
    
    :Reference:
    #. `Jason Weston el at, 2011. WSABIE: Scaling Up To Large Vocabulary Image Annotation <http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf>`
    """
    highest_nout, _ = torch.max(nout, 0)
    return hinge_loss(pout, highest_nout.unsqueeze(-1))


# parser of ranking loss
def margin_ranking_loss_parser(pout  : torch.Tensor, 
                               nout  : torch.Tensor, 
                               target: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""to parse positive outputs, negative outputs and target to a dictionary as a kwargs inputs
    for nn.MarginRankingLoss
    
    Args:
        pout (torch.Tensor), shape = (batch size, 1): scores of positive items
        nout (torch.Tensor)), shape = (batch size, number of negative samples): scores of sampled negative items
        target (torch.Tensor), shape = (batch size, number of negative samples): one likes tensor created by torch.ones_like(nout)
    
    Returns:
        Dict[str, torch.Tensor]: dictionary to be passed to nn.MarginRankingLoss
    
    Key-Values:
        input1: pout, shape = (batch size, 1)
        input2: nout, shape = (batch size, number of negative samples)
        target: target, shape = (batch size, number of negative samples)
    """
    return {"input1": pout, "input2": nout, "target": target}

def soft_margin_loss_parser(pout  : torch.Tensor, 
                            nout  : torch.Tensor, 
                            target: torch.Tensor) -> Dict[str, torch.Tensor]:
    r"""to parse positive outputs, negative outputs and target to a dictionary as a kwargs inputs
    for nn.SoftMarginLoss
    
    Args:
        pout (torch.Tensor), shape = (batch size, 1): scores of positive items
        nout (torch.Tensor)), shape = (batch size, number of negative samples): scores of sampled negative items
        target (torch.Tensor), shape = (batch size, number of negative samples): one likes tensor created by torch.ones_like(nout)
    
    Returns:
        Dict[str, torch.Tensor]: dictionary to be passed to nn.SoftMarginLoss
    
    Key-Values:
        input: (pout - nout), shape = (batch size, number of negative samples)
        target: target, shape = (batch size, number of negative samples)
    """
    return {"input": pout - nout, "target": target}


# groupwise ranking loss
def listnet_loss(yhat: torch.Tensor, ytrue: torch.Tensor) -> torch.Tensor:
    r"""Cross-Entropy implementation in ListNet by the following formula:
    :math:`loss = \sum \text{Softmax} (y_{true}) * \text{log} (\text{Softmax} (\^{y}))`_.
    
    Args:
        yhat (torch.Tensor), shape = (batch size, sequence len), dtype = torch.float: Predicted Ranking scores
        ytrue (torch.Tensor), shape = (batch size, sequence len ), dtype = torch.float: True Ranking scores, e.g. [Excellent(4), Perfect(3), Good(2), Fair(1), Bad(0)] 
    
    Returns:
        torch.Tensor: cross-entropy loss 
    
    :Reference:
    #. `Zhe Cao et al, 2007. Learning to Rank: From Pairwise Approach to Listwise Approach <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf>` 
    """
    # calculate softmax for each row of sample
    phat = torch.softmax(yhat, dim=0)
    ptrue = torch.softmax(ytrue, dim=0)

    # calculate loss
    loss = - (ptrue * phat.log()).sum()

    return loss
