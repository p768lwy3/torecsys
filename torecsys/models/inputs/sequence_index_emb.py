from . import _Inputs
import torch

class SequenceIndexEmbedding(_Inputs):
    def __init__(self):
        super(SequenceIndexEmbedding, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
