from . import _Inputs
import torch

class TextInputs(_Inputs):
    def __init__(self):
        super(TextInputs, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
