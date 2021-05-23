import torch

from torecsys.layers import GeneralizedMatrixFactorizationLayer
from torecsys.models.emb import EmbBaseModel


class MatrixFactorizationModel(EmbBaseModel):
    """
    Model class of Matrix Factorization (MF).
    
    Matrix Factorization is to embed relations between paris of data, like user and item.

    """

    def __init__(self):
        """
        Initialize MatrixFactorizationModel
        """
        super().__init__()
        self.mf = GeneralizedMatrixFactorizationLayer()

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of MatrixFactorizationModel
        
        Args:
            emb_inputs (T), shape = (B, 2, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, 1), data_type = torch.float: output of MatrixFactorizationModel
        """
        # Calculate with mf layer forwardly
        # inputs: emb_inputs, shape = (B, N = 2, E)
        # output: outputs, shape = (B, O = 1)
        outputs = self.mf(emb_inputs)

        # Drop names of outputs, since autograd doesn't support NamedTensor yet
        outputs = outputs.rename(None)

        return outputs

    def predict(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplemented
