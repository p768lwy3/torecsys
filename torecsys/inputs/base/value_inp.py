from typing import Callable, Optional

import torch

from torecsys.inputs.base import BaseInput


class ValueInput(BaseInput):
    """
    Base Input class for value to be passed directly.
    """

    def __init__(self, num_fields: int, transforms: Optional[Callable] = None):
        """
        Initialize ValueInput
        
        Args:
            num_fields (int): Number of inputs' fields.
        """
        super().__init__()

        self.num_fields = num_fields
        self.transforms = transforms
        self.length = 1

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of ValueInput.
        
        Args:
            inputs (T), shape = (B, N): Tensor of values in input fields.
        
        Returns:
            T, shape = (B, 1, N): Outputs of ValueInput
        """
        if inputs.dim() == 2:
            inputs = inputs.unsqueeze(dim=-1)

        if self.transforms:
            inputs = self.transforms(inputs)

        inputs.names = ('B', 'N', 'E',)

        return inputs
