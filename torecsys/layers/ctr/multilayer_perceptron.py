from typing import List, Optional, Dict, Tuple

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class MultilayerPerceptionLayer(BaseLayer):
    """
    Layer class of Multilayer Perception (MLP), which is also called fully connected layer, dense layer,
    deep neural network, etc, to calculate high order non linear relations of features with a stack of linear,
    dropout and activation.
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
                 layer_sizes: List[int],
                 dropout_p: Optional[List[float]] = None,
                 activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize MultilayerPerceptionLayer
        
        Args:
            inputs_size (int): input size of MLP, i.e. size of embedding tensor
            output_size (int): output size of MLP
            layer_sizes (List[int]): layer sizes of MLP
            dropout_p (List[float], optional): probability of Dropout in MLP. Defaults to None
            activation (torch.nn.Module, optional): activation function in MLP. Defaults to nn.ReLU()
        """
        super().__init__()

        if dropout_p is not None and len(dropout_p) != len(layer_sizes):
            raise ValueError('length of dropout_p must be equal to length of layer_sizes.')

        self.embed_size = inputs_size

        layer_sizes = [inputs_size] + layer_sizes

        self.model = nn.Sequential()
        for i, (in_f, out_f) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.model.add_module(f'Linear_{i}', nn.Linear(in_f, out_f))
            if activation is not None:
                self.model.add_module(f'Activation_{i}', activation)
            if dropout_p is not None:
                self.model.add_module(f'Dropout_{i}', nn.Dropout(dropout_p[i]))

        self.model.add_module('LinearOutput', nn.Linear(layer_sizes[-1], output_size))

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of MultilayerPerceptionLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N, O), data_type = torch.float: output of MLP
        """
        # Calculate with model forwardly
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, O)
        outputs = self.model(emb_inputs.rename(None))

        # Rename tensor names
        if outputs.dim() == 2:
            outputs.names = ('B', 'O',)
        elif outputs.dim() == 3:
            outputs.names = ('B', 'N', 'O',)

        return outputs
