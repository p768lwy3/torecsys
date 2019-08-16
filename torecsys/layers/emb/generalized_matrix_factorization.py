from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn


class GeneralizedMatrixFactorizationLayer(nn.Module):
    r"""GeneralizedMatrixFactorization is a layer generalized matrix factorization in a linear 
    format, and used in Neural Collaborative Filtering.
    
    Reference:

    #. `Xiangnan He et al, 2017. Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_.
    
    """
    @jit_experimental
    def __init__(self):
        r"""initialize generalized matrix factorization layer module
        """
        super(GeneralizedMatrixFactorizationLayer, self).__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed-forward calculation of generalized matrix factorization
        
        Args:
            inputs (torch.Tensor), shape = (B, 2, E), dtype=torch.float: inputs of two features vectors
        
        Returns:
            torch.Tensor, shape = (B, 1), dtype = torch.float: output of generalized matrix factorization
        """
        outputs = (inputs[:, 0, :] * inputs[:, 1, :]).sum(dim=1, keepdim=True)
        return outputs.unsqueeze(1)
