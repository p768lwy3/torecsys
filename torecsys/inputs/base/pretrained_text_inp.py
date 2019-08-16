from . import _Inputs
import torch

class PretrainedTextInputs(_Inputs):
    def __init__(self):
        super(PretrainedTextInputs, self).__init__()
        raise NotImplementedError("")
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
