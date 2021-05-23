from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import InnerProductNetworkLayer, OuterProductNetworkLayer, DNNLayer
from torecsys.models.ctr import CtrBaseModel
from torecsys.utils.operations import combination


class ProductNeuralNetworkModel(CtrBaseModel):
    """
    Model class of Product Neural Network (PNN)

    Product Neural Network is a model using inner-product or outer-product to extract high dimensional non-linear
    relationship from interactions of feature tensors instead, where the process is handled by factorization machine
    part in Factorization-machine supported Neural Network (FNN).

    :Reference:

    #. `Yanru QU, 2016. Product-based Neural Networks for User Response Prediction <https://arxiv.org/abs/1611.00144>`_.

    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 deep_layer_sizes: List[int],
                 output_size: int = 1,
                 prod_method: str = 'inner',
                 use_bias: Optional[bool] = True,
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU(),
                 **kwargs):
        """
        Initialize ProductNeuralNetworkModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            deep_layer_sizes (List[int]): layer sizes of dense network
            output_size (int): output size of model. i.e. output size of dense network. Defaults to 1
            prod_method (str): method of product neural network. Allow: ["inner", "outer"]. Defaults to inner
            use_bias (bool, optional): whether the bias constant is concatenated to the input. Defaults to True
            deep_dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            deep_activation (torch.nn.Module, optional): activation function of dense network. Defaults to nn.ReLU()
        
        Arguments:
            kernel_type (str): type of kernel to compress outer-product.
        """
        super().__init__()

        if prod_method == 'inner':
            self.pnn = InnerProductNetworkLayer(num_fields=num_fields)
        elif prod_method == 'outer':
            self.pnn = OuterProductNetworkLayer(embed_size=embed_size,
                                                num_fields=num_fields,
                                                kernel_type=kwargs.get('kernel_type', 'mat'))
        else:
            raise ValueError(f'{prod_method} is not allowed in prod_method. Required: ["inner", "outer"].')

        self.use_bias = use_bias

        cat_size = combination(num_fields, 2) + num_fields
        if self.use_bias:
            cat_size += 1
        self.deep = DNNLayer(
            output_size=output_size,
            layer_sizes=deep_layer_sizes,
            inputs_size=cat_size,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )

        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros((1, 1), names=('B', 'O',)))
            nn.init.uniform_(self.bias.data)

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of ProductNeuralNetworkModel
        
        Args:
            feat_inputs (T), shape = (B, N, E = 1), data_type = torch.float: features tensors
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of ProductNeuralNetworkModel
        """
        # Name the feat_inputs tensor for flatten
        feat_inputs.names = ('B', 'N', 'E',)

        # Aggregate feat_inputs on dimension N and rename dimension O to E
        # inputs: feat_inputs, shape = (B, N, E = 1)
        # output: pnn_first, shape = (B, O = N)
        pnn_first = feat_inputs.flatten(('N', 'E',), 'O')

        # Calculate product cross features by pnn layer 
        # inputs: emb_inputs, shape = (B, N, E)
        # with output's shape = (B, NC2)
        pnn_second = self.pnn(emb_inputs)

        # Concat pnn_second, pnn_first and bias on dimension O
        # inputs: pnn_second, shape = (B, O = NC2)
        # inputs: pnn_first, shape = (B, O = N)
        # inputs: bias, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        pnn_outputs = [pnn_second, pnn_first]
        if self.use_bias:
            batch_size = feat_inputs.size('B')
            bias = self.bias.rename(None).repeat(batch_size, 1)
            bias.names = ('B', 'O',)
            pnn_outputs.append(bias)
        outputs = torch.cat(pnn_outputs, dim='O')

        # Calculate with deep layer forwardly
        # inputs: outputs, shape = (B, O = (NC2 + N + 1))
        # output: outputs, shape = (B, O)
        outputs = self.deep(outputs)

        # Drop names of outputs, since autograd doesn't support NamedTensor yet
        outputs = outputs.rename(None)

        return outputs
