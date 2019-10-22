from . import _CtrModel
import torch
import torch.nn as nn
from torecsys.layers import CrossNetworkLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Callable, List


class DeepAndCrossNetworkModel(_CtrModel):
    r"""DeepAndCrossNetworkModel is a model of deep and cross network, which is a model of 
    a concatenation of deep neural network and cross network, and finally pass to a fully 
    connect layer for the output.

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
        r"""initialize deep adn cross network
        
        Args:
            inputs_size (int): size of inputs tensor
            deep_output_size (int): size of outputs of deep neural network
            deep_layer_sizes (List[int]): sizes of layers in deep neural network
            cross_num_layers (int): number of layers in cross network
            output_size (int, optional): output size of model, i.e. output size of the last 
                dense layer. Defaults to 1.
            deep_dropout_p (List[float], optional): dropout probability for each dense layer. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): activation function for each dense 
                layer. Defaults to nn.ReLU().
        
        Attributes:
            deep (nn.Module): module of dense layer
            cross (nn.Module): module of cross network layer
            fc (nn.Module): module of linear layer to project the output to output size
        """
        super(DeepAndCrossNetworkModel, self).__init__()

        # initialize the layers of module
        # 1. deep output's shape = (B, N, O_d = deep_output_size)
        self.deep = DNNLayer(
            inputs_size = inputs_size, 
            output_size = deep_output_size, 
            layer_sizes = deep_layer_sizes, 
            dropout_p   = deep_dropout_p, 
            activation  = deep_activation
        )
        
        # 2. cross output's shape = (B, O_c = E)
        self.cross = CrossNetworkLayer(
            inputs_size = inputs_size,
            num_layers  = cross_num_layers
        )

        # initialize output fc layer, with output shape = (B, O = output_size)
        cat_size = deep_output_size + inputs_size
        self.fc = nn.Linear(cat_size, output_size)
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of deep and cross network
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: second order terms of fields 
                that will be passed into afm layer and can be get from nn.Embedding(embed_size=E).
        
        Returns:
            T, shape = (B, O), dtype = torch.float: output of deep and cross network
        """
        # inputs' shape = (B, N, E) and reshape to (B, N * E)
        emb_inputs = emb_inputs.flatten(["N", "E"], "E")

        # deep_out's shape = (B, O_d)
        deep_out = self.deep(emb_inputs)

        # cross_out's shape = (B, O_c)
        cross_out = self.cross(emb_inputs)
        
        # cat in third dimension and return shape = (B, O_d + O_c)
        outputs = torch.cat([cross_out, deep_out], dim="O")

        # pass outputs to fully-connect layer, and return shape = (B, O)
        outputs = self.fc(outputs)
        outputs.names = ("B", "O")

        # since autograd does not support Named Tensor at this stage,
        # drop the name of output tensor.
        outputs = outputs.rename(None)
        
        return outputs
