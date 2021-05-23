from typing import Dict, Tuple

import torch

from torecsys.layers import BaseLayer


class InnerProductNetworkLayer(BaseLayer):
    """
    Layer class of Inner Product Network.
    
    Inner Product Network is an option in Product based Neural Network by Yanru Qu et at, 2016, by calculating inner
    product between embedded tensors element-wisely to get cross features interactions
    
    :Reference:

    #. `Yanru Qu et at, 2016. Product-based Neural Networks for User Response Prediction
    <https://arxiv.org/abs/1611.00144>`_.
    
    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'N', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'NC2',)
        }

    def __init__(self,
                 num_fields: int):
        """
        Initialize InnerProductNetworkLayer
        
        Args:
            num_fields (int): number of inputs' fields
        """
        super().__init__()

        # create row idx and col idx to index inputs for inner product
        row_idx = []
        col_idx = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row_idx.append(i)
                col_idx.append(j)
        self.row_idx = torch.LongTensor(row_idx)
        self.col_idx = torch.LongTensor(col_idx)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of InnerProductNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors.
        
        Returns:
            T, shape = (B, NC2), data_type = torch.float: output of InnerProductNetworkLayer
        """
        # calculate inner product between each field
        # inputs: emb_inputs, shape = (B, N, E)
        # output: inner, shape = (B, NC2, E)
        emb_inputs = emb_inputs.rename(None)
        inner = emb_inputs[:, self.row_idx] * emb_inputs[:, self.col_idx]
        inner.names = ('B', 'N', 'E',)

        # aggregate on dimension E
        # inputs: inner, shape = (B, NC2, E)
        # output: outputs, shape = (B, NC2)
        outputs = torch.sum(inner, dim='E')

        # rename tensor names
        outputs.names = ('B', 'O')

        return outputs
