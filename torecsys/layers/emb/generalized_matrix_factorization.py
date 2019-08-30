from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn


class GeneralizedMatrixFactorizationLayer(nn.Module):
    r"""Layer class of Matrix Factorization (MF) to calculate matrix factorzation in a general 
    linear format, which is used in Neural Collaborative Filtering to calculate dot product between 
    user tensors and items tensors.
    
    Reference:

    #. `Xiangnan He et al, 2017. Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_.
    
    """
    @jit_experimental
    def __init__(self):
        r"""Initialize GeneralizedMatrixFactorizationLayer
        """
        # refer to parent class
        super(GeneralizedMatrixFactorizationLayer, self).__init__()

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of GeneralizedMatrixFactorizationLayer
        
        Args:
            emb_inputs (T), shape = (B, 2, E), dtype = torch.float: Embedded features tensors of users and items.
        
        Returns:
            T, shape = (B, 1, 1), dtype = torch.float: Output of GeneralizedMatrixFactorizationLayer.
        """
        # calculate dot product between user tensors and item tensors
        # outputs' shape = (B, 1)
        outputs = (emb_inputs[:, 0, :] * emb_inputs[:, 1, :]).sum(dim=1, keepdim=True)

        # unsqueeze(1) to transform outputs' shape to (B, 1, 1)
        return outputs.unsqueeze(1)
