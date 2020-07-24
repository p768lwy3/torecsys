import torch
import torch.nn as nn

from torecsys.utils.decorator import no_jit_experimental_by_namedtensor


class GeneralizedMatrixFactorizationLayer(nn.Module):
    r"""Layer class of Matrix Factorization (MF).
    
    Matrix Factorization is to calculate matrix factorization in a general linear format,
    which is used in Neural Collaborative Filtering to calculate dot product between user 
    tensors and items tensors.
    
    Reference:

    #. `Xiangnan He et al, 2017. Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_.
    
    """

    @no_jit_experimental_by_namedtensor
    def __init__(self):
        r"""Initialize GeneralizedMatrixFactorizationLayer
        """
        # Refer to parent class
        super(GeneralizedMatrixFactorizationLayer, self).__init__()

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of GeneralizedMatrixFactorizationLayer
        
        Args:
            emb_inputs (T), shape = (B, 2, E), dtype = torch.float: Embedded features tensors 
                of users and items.
        
        Returns:
            T, shape = (B, 1), dtype = torch.float: Output of GeneralizedMatrixFactorizationLayer.
        """
        # Calculate dot product between tensors of user and item
        # inputs: emb_inputs, shape = (B, 2, E)
        # output: outputs, shape = (B, 1)
        outputs = (emb_inputs[:, 0, :] * emb_inputs[:, 1, :]).sum(dim="E", keepdim=True)

        # Rename tensor names
        outputs.names = ("B", "O")

        return outputs
