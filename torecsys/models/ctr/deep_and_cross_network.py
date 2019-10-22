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
        # refer to parent class
        super(DeepAndCrossNetworkModel, self).__init__()

        # initialize dense layer, and the output's shape = (B, N, O_d = deep_output_size)
        self.deep = DNNLayer(
            inputs_size = inputs_size, 
            output_size = deep_output_size, 
            layer_sizes = deep_layer_sizes, 
            dropout_p   = deep_dropout_p, 
            activation  = deep_activation
        )
        
        # initialize cross layer, and the output's shape = (B, O_c = inputs_size)
        self.cross = CrossNetworkLayer(
            inputs_size = inputs_size,
            num_layers  = cross_num_layers
        )

        # initialize output fc layer, with output shape = (B, O = output_size)
        cat_size = deep_output_size + inputs_size
        self.fc = nn.Linear(cat_size, output_size)
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of DeepAndCrossNetworkModel
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of DeepAndCrossNetworkModel.
        """
        # emb_inputs' shape = (B, N, E) and flatten to (B, N * E)
        emb_inputs = emb_inputs.flatten(["N", "E"], "E")

        # return deep_out with shape = (B, O_d)
        deep_out = self.deep(emb_inputs)

        # return cross_out with shape = (B, O_c)
        cross_out = self.cross(emb_inputs)
        
        # concat on dimension = O, which return shape = (B, O_d + O_c)
        outputs = torch.cat([cross_out, deep_out], dim="O")

        # project to output, which return shape = (B, O)
        outputs = self.fc(outputs)
        outputs.names = ("B", "O")

        # since autograd does not support Named Tensor at this stage,
        # drop the name of output tensor.
        outputs = outputs.rename(None)
        
        return outputs
