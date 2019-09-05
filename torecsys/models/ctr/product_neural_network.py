from . import _CtrModel
from torecsys.layers import InnerProductNetworkLayer, OuterProductNetworkLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental
from torecsys.utils.utils import combination
import torch
import torch.nn as nn
from typing import Callable, List

class ProductNeuralNetworkModel(_CtrModel):
    r"""
    """

    def __init__(self,
                 embed_size       : int,
                 num_fields       : int,
                 output_size      : int,
                 prod_method      : str,
                 deep_layer_sizes : List[int],
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 **kwargs):
        r"""Initialize ProductNeuralNetworkModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            output_size (int): Output size of model
            prod_method (str): Method of product neural network. 
                Allow: [inner, outer].
            deep_layer_sizes (List[int]): Layer sizes of DNN
            deep_dropout_p (List[float], optional): Probability of Dropout in DNN. 
                Allow: [None, list of float for each layer]. 
                Defaults to None.
            deep_activation (Callable[[torch.Tensor], torch.Tensor], optional): Activation function of Linear. 
                Allow: [None, Callable[[T], T]]. 
                Defaults to nn.ReLU().
        
        Raises:
            ValueError: when prod_method is not in [inner, outer].
        """
        # refer to parent class
        super(ProductNeuralNetworkModel, self).__init__()

        # initialize product network
        if prod_method == "inner":
            self.pnn = InnerProductNetworkLayer(num_fields=num_fields)
        elif prod_method == "outer":
            self.pnn = OuterProductNetworkLayer(embed_size=embed_size, 
                                                num_fields=num_fields, 
                                                kernel_type=kwargs.get("kernel_type", "mat"))
        else:
            raise ValueError("%s is not allowe in prod_method. Please use ['inner', 'outer'].")
        
        # calculate size of inputs of DNNLayer
        cat_size = 1 + num_fields + combination(num_fields, 2)

        # initialize dnn layer
        self.dnn = DNNLayer(
            output_size = output_size,
            layer_sizes = deep_layer_sizes,
            inputs_size = cat_size,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        )

        # initialize bias variable
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.uniform_(self.bias.data)

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """Forward calculation of ProductNeuralNetworkModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: Features tensors.
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, 1), dtype = torch.float: Output of ProductNeuralNetworkModel
        """
        # get batch size from feat_inputs
        batch_size = feat_inputs.size(0)

        # calculate product cross features with output's shape = (B, 1, NC2)
        pnn_out = self.pnn(emb_inputs)

        # cat product terms, linear terms, and bias term
        outputs = torch.cat([self.bias.repeat(batch_size, 1, 1), feat_inputs, pnn_out], dim=2)

        # feed forward to dnn layer
        outputs = self.dnn(outputs)

        return outputs
