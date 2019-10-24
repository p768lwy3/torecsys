from . import _CtrModel
import torch
import torch.nn as nn
from torecsys.layers import CrossNetworkLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Callable, List


class DeepAndCrossNetworkModel(_CtrModel):
    r"""Model class of Deep & Cross Network (DCN), which is a concatenation of dense network 
    (deep part) and cross network (cross part).

    :Reference:

    #. `Ruoxi Wang et al, 2017. Deep & Cross Network for Ad Click Predictions <https://arxiv.org/abs/1708.05123>`_.
    
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 inputs_size      : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 cross_num_layers : int,
                 output_size      : int = 1,
                 deep_dropout_p   : List[float] = None, 
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize DeepAndCrossNetworkModel
        
        Args:
            inputs_size (int): Inputs size of dense network and cross network, 
                i.e. number of fields * embedding size.
            deep_output_size (int): Output size of dense network
            deep_layer_sizes (List[int]): Layer sizes of dense network
            cross_num_layers (int): Number of layers of Cross Network
            output_size (int, optional): Output size of model, 
                i.e. output size of the projection layer. 
                Defaults to 1.
            deep_dropout_p (List[float], optional): Probability of Dropout in dense network. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of dense network.
                Defaults to nn.ReLU().
        
        Attributes:
            deep (nn.Module): Module of dense layer.
            cross (nn.Module): Module of cross network layer.
            fc (nn.Module): Module of projection layer, i.e. linear layer of output.
        """
        # Refer to parent class
        super(DeepAndCrossNetworkModel, self).__init__()

        # Initialize dense layer
        self.deep = DNNLayer(
            inputs_size = inputs_size, 
            output_size = deep_output_size, 
            layer_sizes = deep_layer_sizes, 
            dropout_p   = deep_dropout_p, 
            activation  = deep_activation
        )
        
        # Initialize cross layer
        self.cross = CrossNetworkLayer(
            inputs_size = inputs_size,
            num_layers  = cross_num_layers
        )

        # Initialize linear layer
        cat_size = deep_output_size + inputs_size
        self.fc = nn.Linear(cat_size, output_size)
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of DeepAndCrossNetworkModel
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of DeepAndCrossNetworkModel.
        """
        # Flatten emb_inputs
        # inputs: emb_inputs, shape = (B, N, E)
        # output: emb_inputs, shape = (B, N * E)
        emb_inputs = emb_inputs.flatten(["N", "E"], "E")

        # Calculate with cross layer forwardly
        # inputs: emb_inputs, shape = (B, N * E)
        # output: cross_out, shape = (B, O = Oc)
        cross_out = self.cross(emb_inputs)

        # Calculate with dense layer forwardly
        # inputs: emb_inputs, shape = (B, N * E)
        # output: deep_out, shape = (B, O = Od)
        deep_out = self.deep(emb_inputs)
        
        # Concatenate on dimension = O,
        # inputs: cross_out, shape = (B, Oc)
        # inputs: deep_out, shape = (B, Od)
        # output: outputs, shape = (B, O = Od + Oc)
        outputs = torch.cat([cross_out, deep_out], dim="O")

        # Calculate with linear layer forwardly
        # inputs: outputs, shape = (B, O = Od + Oc)
        # output: outputs, shape = (B, O = Ofc)
        outputs = self.fc(outputs)
        outputs.names = ("B", "O")

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)
        
        return outputs
