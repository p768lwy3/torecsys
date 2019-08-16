from . import _EmbModel
from torecsys.layers import GeneralizedMatrixFactorizationLayer
from torecsys.utils.decorator import jit_experimental
import torch


class MatrixFactorizationModel(_EmbModel):
    r"""MatrixFactorizationModel is a model of matrix factorization to embed relations between paris of data
    """
    def __init__(self):
        r"""initialize matrix factorization model to embed index to vectors
        """
        super(MatrixFactorizationModel, self).__init__()
        self.mf = GeneralizedMatrixFactorizationLayer()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed-forward calculation of matrix factorization
        
        Args:
            inputs (torch.Tensor), shape = (B, 2, E), dtype = torch.float: inputs of two features vectors
        
        Returns:
            torch.Tensor, shape = (B, 1), dtype = torch.float: scores of matrix factorization model
        """
        outputs = self.mf(inputs)
        return outputs
