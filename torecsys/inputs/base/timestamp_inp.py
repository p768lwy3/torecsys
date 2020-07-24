import torch

from . import Inputs


class TimestampInputs(Inputs):
    r"""Base Inputs class for Timestamp.
    """

    def __init__(self):
        super(TimestampInputs, self).__init__()
        raise NotImplementedError("")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
