import torch

from . import Inputs


class AudioInputs(Inputs):
    r"""Base Inputs class for Audio.
    """

    def __init__(self):
        super(AudioInputs, self).__init__()
        raise NotImplementedError("")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
