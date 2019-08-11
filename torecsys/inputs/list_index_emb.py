from . import _Inputs
import torch

class ListIndexEmbedding(_Inputs):
    def __init__(self):
        super(ListIndexEmbedding, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
