import torch

from . import Inputs


class TextInputs(Inputs):
    r"""Base Inputs class for Text.
    """

    def __init__(self):
        super(TextInputs, self).__init__()
        raise NotImplementedError("")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
