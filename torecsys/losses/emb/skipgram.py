import torch

from torecsys.losses.emb import EmbLoss
from torecsys.losses.emb.functional import skip_gram_loss


class SkipGramLoss(EmbLoss):
    r"""SkipGramLoss is a model to calculate the loss used in SkipGram algorithm
    :title:`Tomas Mikolov et al, 2013`[1] which is to calculate the loss by the following formula:
    :math:`loss = - \sum_{c=1}^{C} u_{j_{c}^{*}} + C log ( \sum_{j'=1}^{v} e^{u_{j'}} )` .
    
    :Reference:

    #. `Tomas Mikolov et al, 2013. Efficient Estimation of Word Representations in Vector Space
    <https://arxiv.org/abs/1301.3781>`_.
    
    """

    def __init__(self):
        """
        Initialize SkipGramLoss
        """
        super().__init__()

    @staticmethod
    def forward(content_inputs: torch.Tensor,
                pos_inputs: torch.Tensor,
                neg_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of SkipGramLoss

        Args:
            content_inputs (T), shape = (B, 1, E), data_type = torch.float: Predicted scores of content or anchor.
            pos_inputs (T), shape = (B, 1, E), data_type = torch.float: Predicted scores of positive samples.
            neg_inputs (T, shape = (B, N Neg, E), data_type = torch.float: Predicted scores of negative samples.

        Returns:
            T, shape = (1, ), data_type = torch.float: Output of SkipGramLoss.
        """
        # Calculate skip gram loss
        loss = skip_gram_loss(content_inputs, pos_inputs, neg_inputs)
        return loss
