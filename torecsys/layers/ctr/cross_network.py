from typing import Dict, Tuple

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class CrossNetworkLayer(BaseLayer):
    r"""
    Layer class of Cross Network.
    
    Cross Network was used in Deep & Cross Network, to calculate cross features interaction between element,
    by the following equation: for i-th layer, :math:`x_{i} = x_{0} * (w_{i} * x_{i-1} + b_{i}) + x_{0}`.

    :Reference:

    #. `Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`
    
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
        Initialize CrossNetworkLayer
        
        Args:
            inputs_size (int): inputs size of Cross Network, i.e. size of embedding tensor
            num_layers (int): number of layers of Cross Network
        """
        super().__init__()

        self.embed_size = inputs_size

        self.model = nn.ModuleList()
        for _ in range(num_layers):
            self.model.append(nn.Linear(inputs_size, inputs_size))

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of CrossNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N, E), data_type = torch.float: output of CrossNetworkLayer
        """
        # deep copy emb_inputs to outputs as residual
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, E = O)
        outputs = emb_inputs.detach().requires_grad_()

        # drop names since einsum doesn't support NamedTensor now
        emb_inputs.names = None
        outputs.names = None

        # calculate with linear forwardly and add residual to outputs
        # inputs: emb_inputs, shape = (B, N, E)
        # inputs: outputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, E)
        for layer in self.model:
            # outputs = emb_inputs * layer(outputs) + emb_inputs
            outputs = layer(outputs)
            outputs = torch.einsum('ijk,ijk->ijk', emb_inputs, outputs)
            outputs = outputs + emb_inputs

        # rename tensor names
        if outputs.dim() == 2:
            outputs.names = ('B', 'O',)
        elif outputs.dim() == 3:
            outputs.names = ('B', 'N', 'O',)

        return outputs
