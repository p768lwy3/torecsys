"""
torecsys.losses.emb.functional is a sub model of functions for the implementation of losses in
embedding.
"""

import torch
import torch.nn.functional as F


def skip_gram_loss(cout: torch.Tensor, pout: torch.Tensor, nout: torch.Tensor) -> torch.Tensor:
    r"""
    Calculation of loss of skip gram, an algorithm used in Word2Vec, which is to calculate
    loss by the following formula: :math:`loss = - \sum_{c=1}^{C} u_{j_{c}^{*}} + C log ( \sum_{j'=1}^{v} e^{u_{j'}} )`.
    
    Args:
        cout (T), shape = (B, 1, E), data_type = torch.float: predicted scores of content or anchor
        pout (T), shape = (B, 1, E), data_type = torch.float: predicted scores of positive samples
        nout (T), shape = (B, 1, E), data_type = torch.float: predicted scores of negative samples

    Returns:
        T, shape = (1, ), data_type = torch.float: output of skip_gram_loss
    """
    # positive values' part, return shape = (B, )
    pval = F.logsigmoid((cout * pout).sum(dim=2)).squeeze()

    # negative values' part, bmm (B, num neg, E) 
    # by (B, E, 1). Hence, shape = (B, num nge, 1)
    nval = torch.bmm(nout, cout.transpose(1, 2))

    # sum by second dimension, and hence return shape = (B, 1, 1)
    # take log sigmoid and squeeze, then return shape = (B, )
    nval = F.logsigmoid(- nval.sum(dim=1)).squeeze()

    # calculate loss and take aggregation
    loss = - (pval + nval).mean()

    return loss
