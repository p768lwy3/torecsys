from typing import Dict, Tuple

import torch

from torecsys.layers import BaseLayer


class GeneralizedMatrixFactorizationLayer(BaseLayer):
    """
    Layer class of Matrix Factorization (MF).
    
    Matrix Factorization is to calculate matrix factorization in a general linear format, which is used in
    Neural Collaborative Filtering to calculate dot product between user tensors and items tensors.
    
    Reference:

    #. `Xiangnan He et al, 2017. Neural Collaborative Filtering <https://arxiv.org/abs/1708.05031>`_.
    
    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', '2', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', '1',)
        }

    def __init__(self):
        """
        Initialize GeneralizedMatrixFactorizationLayer
        """
        super().__init__()

    @staticmethod
    def forward(emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of GeneralizedMatrixFactorizationLayer
        
        Args:
            emb_inputs (T), shape = (B, 2, E), data_type = torch.float: embedded features tensors
                of users and items.
        
        Returns:
            T, shape = (B, 1), data_type = torch.float: output of GeneralizedMatrixFactorizationLayer
        """
        # Name the inputs tensor for alignment
        emb_inputs.names = ('B', 'N', 'E',)

        # Calculate dot product between tensors of user and item
        # inputs: emb_inputs, shape = (B, 2, E)
        # output: outputs, shape = (B, 1)
        outputs = (emb_inputs[:, 0, :] * emb_inputs[:, 1, :]).sum(dim='E', keepdim=True)

        # Rename tensor names
        outputs.names = ('B', 'O')

        return outputs
