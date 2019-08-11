from . import _Inputs
import torch

class PretrainedImageInputs(_Inputs):
    def __init__(self):
        super(PretrainedImageInputs, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
