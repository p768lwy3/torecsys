from typing import Tuple, Dict

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class AttentionalFactorizationMachineLayer(BaseLayer):
    """
    Layer class of Attentional Factorization Machine (AFM).
    
    Attentional Factorization Machine is to calculate interaction between each pair of features 
    by using element-wise product (i.e. Pairwise Interaction Layer), compressing interaction 
    tensors to a single representation. The output shape is (B, 1, E).

    Attributes:
        row_idx: ...
        col_idx: ...
        attention: ...
        dropout: ...

    References:

    - `Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via
    Attention Networks∗ <https://arxiv.org/abs/1708.04617>`_.

    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        """
        Dict[str, Tuple[str, ...]]: inputs_size of the layer
        """
        return {
            'inputs': ('B', 'N', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        """
        Dict[str, Tuple[str, ...]]: outputs_size of the layer
        """
        return {
            'outputs': ('B', 'E',),
            'attn_scores': ('B', 'NC2', '1',)
        }

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 attn_size: int,
                 dropout_p: float = 0.1):
        """
        Initialize AttentionalFactorizationMachineLayer
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            attn_size (int): size of attention layer
            dropout_p (float, optional): probability of Dropout in AFM
                Defaults to 0.1.
        """
        super().__init__()

        self.row_idx: list = []
        self.col_idx: list = []

        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row_idx.append(i)
                self.col_idx.append(j)

        self.row_idx = torch.LongTensor(self.row_idx)
        self.col_idx = torch.LongTensor(self.col_idx)

        self.attention: nn.Sequential = nn.Sequential()
        self.attention.add_module('Linear', nn.Linear(embed_size, attn_size))
        self.attention.add_module('Activation', nn.ReLU())
        self.attention.add_module('OutProj', nn.Linear(attn_size, 1))
        self.attention.add_module('Softmax', nn.Softmax(dim=1))
        self.attention.add_module('Dropout', nn.Dropout(dropout_p))

        self.dropout: nn.Module = nn.Dropout(dropout_p)

    def forward(self, emb_inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward calculation of AttentionalFactorizationMachineLayer

        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns: Tuple[T], shape = ((B, E) (B, NC2, 1)), data_type = torch.float: output of
            AttentionalFactorizationMachineLayer and Attention weights
        """
        # Calculate hadamard product
        # inputs: emb_inputs, shape = (B, N, E)
        # output: products, shape = (B, NC2, E)
        emb_inputs = emb_inputs.rename(None)
        products = torch.einsum('ijk,ijk->ijk', emb_inputs[:, self.row_idx], emb_inputs[:, self.col_idx])

        # Calculate attention scores
        # inputs: products, shape = (B, NC2, E)
        # output: attn_scores, shape = (B, NC2, 1)
        attn_scores = self.attention(products.rename(None))

        # Apply attention on inner product
        # inputs: products, shape = (B, NC2, E)
        # inputs: attn_scores, shape = (B, NC2, 1)
        # output: outputs, shape = (B, E)
        outputs = torch.einsum('ijk,ijh->ijk', products, attn_scores)
        outputs.names = ('B', 'N', 'E')
        outputs = outputs.sum(dim='N')

        # Apply dropout on outputs
        # inputs: outputs, shape = (B, E)
        # output: outputs, shape = (B, E)
        outputs = self.dropout(outputs)

        return outputs, attn_scores
