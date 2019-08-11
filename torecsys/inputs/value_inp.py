from . import _Inputs
import torch

class ValueInputs(_Inputs):
    def __init__(self):
        super(ValueInputs, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
