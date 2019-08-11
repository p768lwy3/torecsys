from . import _Inputs
import torch

class AudioInputs(_Inputs):
    def __init__(self):
        super(AudioInputs, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
