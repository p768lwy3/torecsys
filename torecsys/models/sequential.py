from typing import Dict

import torch
import torch.nn as nn

from torecsys.inputs import BaseInput


class Sequential(nn.Module):
    """
    Sequential container, where the model of embeddings and model will be stacked in the order they are passed to
    the constructor
    """

    def __init__(self,
                 inputs: BaseInput,
                 model: nn.Module):
        """
        Initialize Sequential container
        
        Args:
            inputs (BaseInput): inputs where the return is a dictionary of inputs' tensors which are passed to
                the model directly
            model (nn.Module): model class to be trained and used in prediction
        """
        super().__init__()

        self._inputs = inputs
        self._model = model

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward calculation of Sequential
        
        Args:
            inputs (Dict[str, T]): dictionary of inputs,
                where key is string of input fields' name, and value is torch.tensor pass to Input class
        
        Returns:
            torch.Tensor: output of model
        """
        inputs = self._inputs(inputs)
        outputs = self._model(**inputs)
        return outputs
