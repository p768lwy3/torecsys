import torch

from torecsys.layers import GeneralizedMatrixFactorizationLayer
from . import _EmbModel


class MatrixFactorizationModel(_EmbModel):
    r"""Model class of Matrix Factorization (MF).
    
    Matrix Factorization is to embed relations between paris of data, like user and item.

    """

    def __init__(self):
        r"""Initialize MatrixFactorizationModel

        Attributes:
            mf (nn.Module): Module of matrix factorization layer
        """
        # Refer to parent class
        super(MatrixFactorizationModel, self).__init__()

        # Initialize mf layer
        self.mf = GeneralizedMatrixFactorizationLayer()

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of MatrixFactorizationModel
        
        Args:
            emb_inputs (T), shape = (B, 2, E), dtype = torch.float: Embedded features tensors
        
        Returns:
            T, shape = (B, 1), dtype = torch.float: Output of MatrixFactorizationModel
        """
        # Calculate with mf layer forwardly
        # inputs: emb_inputs, shape = (B, N = 2, E)
        # output: outputs, shape = (B, O = 1)
        outputs = self.mf(emb_inputs)

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
