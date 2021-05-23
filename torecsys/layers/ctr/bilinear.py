from typing import Dict, Tuple

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class BilinearNetworkLayer(BaseLayer):
    """
    Layer class of Bilinear.
    
    Bilinear is to calculate interaction in element-wise by nn.Bilinear, which the calculation
    is: for i-th layer, :math:`x_{i} = (x_{0} * A_{i} * x_{i - 1}) + b_{i} + x_{0}`, where 
    :math:`A_{i}` is the weight of model of shape :math:`(O_{i}, I_{i1}, I_{i2})`.

    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'N', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', 'N', 'E',)
        }

    def __init__(self,
                 inputs_size: int,
                 num_layers: int):
        """
        Initialize BilinearNetworkLayer

        Args:
            inputs_size (int): input size of Bilinear, i.e. size of embedding tensor
            num_layers (int): number of layers of Bilinear Network
        """
        super().__init__()

        self.model = nn.ModuleList()
        for _ in range(num_layers):
            self.model.append(nn.Bilinear(inputs_size, inputs_size, inputs_size))

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of BilinearNetworkLayer
        
        Args:
            emb_inputs (T), shape = shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N, E), data_type = torch.float: output of BilinearNetworkLayer
        """
        # Deep copy emb_inputs to outputs as residual
        # inputs: emb_inputs, shape = (B, O = N * E)
        # output: outputs, shape = (B, O = N * E)
        outputs = emb_inputs.detach().requires_grad_()

        # Calculate with bilinear forwardly and add residual to outputs
        # inputs: emb_inputs, shape = (B, N, E)
        # inputs: outputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, E)
        for layer in self.model:
            outputs = layer(emb_inputs.rename(None), outputs.rename(None))
            outputs = outputs + emb_inputs

        if outputs.dim() == 2:
            outputs.names = ('B', 'N')
        elif outputs.dim() == 3:
            outputs.names = ('B', 'N', 'O')

        return outputs
