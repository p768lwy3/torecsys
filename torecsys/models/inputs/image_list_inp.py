from . import _Inputs
import torch

class ImageListInputs(_Inputs):
    def __init__(self):
        super(ImageListInputs, self).__init__()
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
