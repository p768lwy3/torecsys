from . import _CtrModel
from torecsys.layers import InnerProductNetworkLayer, OuterProductNetworkLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from torecsys.utils.utils import combination
import torch
import torch.nn as nn
from typing import Callable, List

class ProductNeuralNetworkModel(_CtrModel):
    r"""Model class of Product Neural Network (PNN).

    Product Neural Network is a model using inner-product or outer-product to extract high 
    dimensional non-linear relationship from interactions of feature tensors instead, where 
    the process is handled by factorization machine part in Factorization-machine supported 
    Neural Network (FNN).

    :Reference:

    #. `Yanru QU, 2016. Product-based Neural Networks for User Response Prediction <https://arxiv.org/abs/1611.00144>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size       : int,
                 num_fields       : int,
                 deep_layer_sizes : List[int],
                 output_size      : int = 1,
                 prod_method      : str = "inner", 
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 **kwargs):
        r"""Initialize ProductNeuralNetworkModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            deep_layer_sizes (List[int]): Layer sizes of dense network
            output_size (int): Output size of model
                i.e. output size of dense network. 
                Defaults to 1.
            prod_method (str): Method of product neural network. 
                Allow: [inner, outer].
                Defaults to inner.
            deep_dropout_p (List[float], optional): Probability of Dropout in dense network. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of dense network. 
                Defaults to nn.ReLU().
        
        Arguments:
            kernel_type (str): Type of kernel to compress outer-product.
        
        Attributes:
            pnn (nn.Module): Module of product neural network.
            deep (nn.Module): Module of dense layer.
            bias (nn.Parameter): Parameter of bias of field-aware factorization machine.

        Raises:
            ValueError: when prod_method is not in [inner, outer].
        """
        # Refer to parent class
        super(ProductNeuralNetworkModel, self).__init__()

        # Initialize product network
        if prod_method == "inner":
            self.pnn = InnerProductNetworkLayer(num_fields  = num_fields)
        elif prod_method == "outer":
            self.pnn = OuterProductNetworkLayer(embed_size  = embed_size, 
                                                num_fields  = num_fields, 
                                                kernel_type = kwargs.get("kernel_type", "mat"))
        else:
            raise ValueError("'%s' is not allowed in prod_method. Please use ['inner', 'outer'].")
        
        # Calculate size of inputs of dense layer
        cat_size = combination(num_fields, 2) + num_fields + 1

        # Initialize dense layer
        self.deep = DNNLayer(
            output_size = output_size,
            layer_sizes = deep_layer_sizes,
            inputs_size = cat_size,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        )

        # Initialize bias parameter
        self.bias = nn.Parameter(torch.zeros((1, 1), names = ("B", "O")))
        nn.init.uniform_(self.bias.data)

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """Forward calculation of ProductNeuralNetworkModel
        
        Args:
            feat_inputs (T), shape = (B, N, E = 1), dtype = torch.float: Features tensors.
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of ProductNeuralNetworkModel
        """
        # Get batch size from emb_inputs
        b = emb_inputs.size("B")

        # Aggregate feat_inputs on dimension N and rename dimension O to E
        # inputs: feat_inputs, shape = (B, N, E = 1)
        # output: pnn_first, shape = (B, O = N)
        pnn_first = feat_inputs.flatten(["N", "E"], "O")

        # Calculate product cross features by pnn layer 
        # inputs: emb_inputs, shape = (B, N, E)
        # with output's shape = (B, NC2)
        pnn_second = self.pnn(emb_inputs)

        # Concat pnn_second, pnn_first and bias on dimension O
        # inputs: pnn_second, shape = (B, O = NC2)
        # inputs: pnn_first, shape = (B, O = N)
        # inputs: bias, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = torch.cat([pnn_second, pnn_first, self.bias], dim="O")

        # Calculate with deep layer forwardly
        # inputs: outputs, shape = (B, O = (NC2 + N + 1))
        # output: outputs, shape = (B, O)
        outputs = self.deep(outputs)

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
