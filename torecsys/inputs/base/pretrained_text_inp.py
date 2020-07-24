import torch

from . import Inputs


class PretrainedTextInputs(Inputs):
    r"""Base Inputs class for text, which embed by famous pre-trained model in NLP.
    """

    def __init__(self):
        super(PretrainedTextInputs, self).__init__()
        raise NotImplementedError("")

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.Tensor
