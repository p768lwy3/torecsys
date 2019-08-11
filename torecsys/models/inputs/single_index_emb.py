from . import _Inputs
import torch

class SingleIndexEmbedding(_Inputs):
    def __init__(self):
        super(SingleIndexEmbedding, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
