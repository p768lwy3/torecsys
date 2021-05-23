from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class WideLayer(BaseLayer):
    """
    Layer class of wide
    
    Wide is a stack of linear and dropout, used in calculation of linear relation frequently

    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'N', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', 'N', 'O',)
        }

    def __init__(self,
                 inputs_size: int,
                 output_size: int,
                 dropout_p: Optional[float] = None):
        """
        Initialize WideLayer
        
        Args:
            inputs_size (int): size of inputs, i.e. size of embedding tensor
            output_size (int): output size of wide layer
            dropout_p (float, optional): probability of Dropout in wide layer. Defaults to None
        """
        super().__init__()

        self.embed_size = inputs_size
        self.model = nn.Sequential()
        self.model.add_module('Linear', nn.Linear(inputs_size, output_size))
        if dropout_p is not None:
            self.model.add_module('Dropout', nn.Dropout(dropout_p))

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of WideLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N, E), data_type = torch.float: output of wide layer
        """
        # Calculate with linear forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, E = O)
        outputs = self.model(emb_inputs.rename(None))

        # Rename tensor names
        if outputs.dim() == 2:
            outputs.names = ('B', 'O',)
        elif outputs.dim() == 3:
            outputs.names = ('B', 'N', 'O',)

        return outputs
