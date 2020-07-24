import torch

from . import Inputs


class ImagesListInputs(Inputs):
    r"""Base Inputs class for list of images
    """

    def __init__(self):
        super(ImagesListInputs, self).__init__()
        raise NotImplementedError("")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
