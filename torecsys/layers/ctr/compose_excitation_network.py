from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class ComposeExcitationNetworkLayer(BaseLayer):
    """
    Layer class of Compose Excitation Network (CEN) / Squeeze-and-Excitation Network (SENET).
    
    Compose Excitation Network was used in FAT-Deep :title:`Junlin Zhang et al, 2019`[1] and 
    Squeeze-and-Excitation Network was used in FibiNET :title:`Tongwen Huang et al, 2019`[2]
    
    #. compose field-aware embedded tensors by a 1D convolution with a :math:`1 * 1` kernel
    feature-wisely from a :math:`k * n` tensor of field i into a :math:`k * 1` tensor. 
    
    #. concatenate the tensors and feed them to dense network to calculate attention 
    weights.
    
    #. inputs' tensor are multiplied by attention weights, and return outputs tensor with
    shape = (B, N * N, E).

    :Reference:

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine
    <https://arxiv.org/abs/1905.06336>`_.

    #. `Tongwen Huang et al, 2019. FibiNET: Combining Feature Importance and Bilinear feature Interaction for
    Click-Through Rate Prediction <https://arxiv.org/abs/1905.09433>`_.

    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'N^2', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', 'N^2', 'E',)
        }

    def __init__(self,
                 num_fields: int,
                 reduction: int,
                 squared: Optional[bool] = True,
                 activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize ComposeExcitationNetworkLayer
        
        Args:
            num_fields (int): number of inputs' fields
            reduction (int): size of reduction in dense layer
            activation (torch.nn.Module, optional): activation function in dense layers. Defaults to nn.ReLU()
        """
        super().__init__()

        inputs_num_fields = num_fields ** 2 if squared else num_fields
        reduced_num_fields = inputs_num_fields // reduction

        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential()
        self.fc.add_module('ReductionLinear', nn.Linear(inputs_num_fields, reduced_num_fields))
        self.fc.add_module('ReductionActivation', activation)
        self.fc.add_module('AdditionLinear', nn.Linear(reduced_num_fields, inputs_num_fields))
        self.fc.add_module('AdditionActivation', activation)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of ComposeExcitationNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N^2, E), data_type = torch.float: field aware embedded features tensors
        
        Returns:
            T, shape = (B, N^2, E), data_type = torch.float: output of ComposeExcitationNetworkLayer
        """
        # Pool emb_inputs
        # inputs: emb_inputs, shape = (B, N^2, E)
        # output: pooled_inputs, shape = (B, N^2, 1)
        pooled_inputs = self.pooling(emb_inputs.rename(None))
        pooled_inputs.names = ('B', 'N', 'E',)

        # Flatten pooled_inputs
        # inputs: pooled_inputs, shape = (B, N^2, 1)
        # output: pooled_inputs, shape = (B, N^2)
        pooled_inputs = pooled_inputs.flatten(('N', 'E',), 'N')

        # Calculate attention weight with dense layer forwardly
        # inputs: pooled_inputs, shape = (B, N^2)
        # output: attn_w, shape = (B, N^2)
        attn_w = self.fc(pooled_inputs.rename(None))
        attn_w.names = ('B', 'N',)

        # Unflatten attention weights and apply it to emb_inputs
        # inputs: attn_w, shape = (B, N^2)
        # inputs: emb_inputs, shape = (B, N^2, E)
        # output: outputs, shape = (B, N^2, E)
        attn_w = attn_w.unflatten('N', (('N', attn_w.size('N'),), ('E', 1,),))

        # Multiply attentional weights on field embedding tensors
        outputs = torch.einsum('ijk,ijh->ijk', emb_inputs.rename(None), attn_w.rename(None))
        outputs.names = ('B', 'N', 'E',)

        return outputs
