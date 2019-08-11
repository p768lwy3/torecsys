from . import _Inputs
import torch

class StackedInputs(_Inputs):
    def __init__(self):
        super(StackedInputs, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
