from . import _Inputs
import torch

class EmbeddingDict(_Inputs):
    def __init__(self):
        super(EmbeddingDict, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
