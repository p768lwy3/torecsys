import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Callable

class ComposeExcitationNetworkLayer(nn.Module):
    r"""Layer class of Compose Excitation Network (CEN) / Squeeze-and-Excitation Network (SENET).
    
    Compose Excitation Network was used in FAT-Deep :title:`Junlin Zhang et al, 2019`[1] and 
    Squeeze-and-Excitation Network was used in FibiNET :title:`Tongwen Huang et al, 2019`[2]
    
    #. compose field-aware embedded tensors by a 1D convalution with a :math:`1 * 1` kernel 
    feature-wisely from a :math:`k * n` tensor of field i into a :math:`k * 1` tensor. 
    
    #. concatenate the tensors and feed them to dense network to calculate attention 
    weights.
    
    #. inputs' tensor are multiplied by attention weights, and return outputs tensor with 
    shape = (B, N * N, E).

    :Reference:

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine <https://arxiv.org/abs/1905.06336>`_.

    #. `Tongwen Huang et al, 2019. FibiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction <https://arxiv.org/abs/1905.09433>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 num_fields : int,
                 reduction  : int = 1,
                 activation : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize ComposeExcitationNetworkLayer
        
        Args:
            num_fields (int): Number of inputs' fields. 
            reduction (int, optional): Size of reduction in dense layer. 
                Defaults to 1.
            activation (Callable[[T], T], optional): Activation function in dense layers.
                Defaults to nn.ReLU().
        
        Attributes:
            pooling (torch.nn.Module): Adaptive average pooling layer to compose tensors.
            fc (torch.nn.Sequential): Sequential of linear and activation to calculate weights of 
                attention, which the linear layers are: 
                :math:`[Linear(N^2, \frac{N^2}{reduction}), Linear(\frac{N^2}{reduction}, N^2)]`. 
        """
        # Refer to parent class
        super(ComposeExcitationNetworkLayer, self).__init__()

        # Initialize 1d pooling layer
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Initialize dense layers
        squared_num_fields = num_fields ** 2
        reduced_num_fields = squared_num_fields // reduction

        self.fc = nn.Sequential()
        self.fc.add_module("ReductionLinear", nn.Linear(squared_num_fields, reduced_num_fields))
        self.fc.add_module("ReductionActivation", activation)
        self.fc.add_module("AdditionLinear", nn.Linear(reduced_num_fields, squared_num_fields))
        self.fc.add_module("AdditionActivation", activation)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of ComposeExcitationNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Field aware embedded features tensors.
        
        Returns:
            T, shape = (B, N, E), dtype = torch.float: Output of ComposeExcitationNetworkLayer.
        """
        # Pool emb_inputs
        # inputs: emb_inputs, shape = (B, N, E)
        # output: pooled_inputs, shape = (B, N, 1)
        pooled_inputs = self.pooling(emb_inputs.rename(None))
        pooled_inputs.names = ("B", "N", "E")

        # Flatten pooled_inputs
        # inputs: pooled_inputs, shape = (B, N, 1)
        # output: pooled_inputs, shape = (B, N)
        pooled_inputs = pooled_inputs.flatten(["N", "E"], "N")

        # Calculate attention weight with dense layer fowardly
        # inputs: pooled_inputs, shape = (B, N)
        # output: attn_w, shape = (B, N)
        attn_w = self.fc(pooled_inputs.rename(None))
        attn_w.names = ("B", "N")

        # Unflatten attention weights and apply it to emb_inputs
        # inputs: attn_w, shape = (B, N)
        # inputs: emb_inputs, shape = (B, N, E)
        # output: outputs, shape = (B, N, E)
        attn_w = attn_w.unflatten("N", (("N", attn_w.size("N")), ("E", 1)))
        
        # Multiply attentional weights on field embedding tensors
        ## outputs = emb_inputs * attn_w
        outputs = torch.einsum("ijk,ijh->ijk", [emb_inputs.rename(None), attn_w.rename(None)])
        outputs.names = ("B", "N", "E")

        return outputs
