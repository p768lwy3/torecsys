from . import _CtrModule
from ..layers import FieldAwareFactorizationMachineLayer, MultilayerPerceptronLayer
from torecsys.utils.logging.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Callable, List


class DeepFieldAwareFactorizationMachineModule(_CtrModule):
    r"""DeepFieldAwareFactorizationMachineModule is a module of Deep Field-aware Factorization 
    Machine (DeepFFM) proposed by Yang et al in Tencent Social Ads competition 2017 (they called
    this model Network on Field-aware Factorization Machine - NFFM), and described and rename to
    Deep Field-aware Factorization Machine by Zhang et al in their research (Zhang et al, 2019). 
    The model is a stack of Field Aware Factorization Machine and Multilayer Perceptron, 

    #. First, calculate the interactions of features of second-order features (i.e. embedding \
    matrices in FFM) by inner-product or hadamard product, Hence, let :math:`A` be the feature
    interaction vectors, :math:`A` will be calculate in the following formula: 
    :math:`\text{Inner Product:} A = [v_{1, 2} \bigoplus v_{2, 1}, ..., v_{i, j} \bigoplus v_{j, i}, ..., v_{(n-1), n} \bigoplus v_{n, (n-1)}]
    :math:`\text{OR Hadamard Product:} A = [v_{1, 2} \bigotimes v_{2, 1}, ..., v_{i, j} \bigotimes v_{j, i}, ..., v_{(n-1), n} \bigotimes v_{n, (n-1)}]

    #. Second, pass the matrices :math:`\text{A}` to a Multilayer Percepton, and the forward
    process is:
    :math:`\text{if i = 1,} x_{1} = \text{activation} ( W_{1} A + b_{1} )` 
    :math:`\text{otherwise,} x_{i} = \text{activation} ( W_{i} x_{i - 1} + b_{i})` 

    #. Finally, concatenate the above part and the linear part :math:`x_{linear}, and pass forward to a linear 
    output layer:
    :math:`y(X) = W_{linear} x_{linear} + W_{second_order} x_{l} + b_{output}` .

    :Reference:

    #. Junlin Zhang et al, 2019. ` FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine <https://arxiv.org/abs/1905.06336>`_.

    """
    def __init__(self,
                 embed_size       : int,
                 num_fields       : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 ffm_dropout_p    : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 output_size      : int = 1):
        r"""initialize Deep Field-aware Factorization Machine Module
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            deep_output_size (int): output size of multilayer perceptron layer
            deep_layer_sizes (List[int]): layer sizes of multilayer perceptron layer
            ffm_dropout_p (float, optional): dropout probability after field-aware factorization machine. Defaults to 0.0.
            deep_dropout_p (List[float], optional): dropout probability after activation of each layer. Allow: [None, list of float for each layer]. Defaults to None.
            deep_activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function of each layer. Allow: [None, Callable[[torch.Tensor], torch.Tensor]]. Defaults to nn.ReLU().
            output_size (int, optional): output size of linear transformation after concatenate. Defaults to 1.
        """
        # initialize nn.Module class
        super(DeepFieldAwareFactorizationMachineModule, self).__init__()

        # sequential of second-order part in inputs
        self.second_order = nn.Sequential()
        # ffm's input shape = (batch size, num_fields * num_fields, embedding size)
        # ffm's output shape = (batch size, num_fields, embedding size)
        self.second_order.add_module("ffm", FieldAwareFactorizationMachineLayer(
            num_fields=num_fields, 
            dropout_p=ffm_dropout_p
        ))
        # deep's input shape = (batch size, num_fields, embedding size)
        # deep's output shape = (batch size, 1, output size)
        self.second_order.add_module("deep", MultilayerPerceptronLayer(
            output_size=deep_output_size, 
            layer_sizes=deep_layer_sizes, 
            embed_size=embed_size, 
            num_fields=num_fields, 
            dropout_p=deep_dropout_p, 
            activation=deep_activation
        ))

        # fully connected layers' input = (batch size, 1, number of fields + output size)
        # and output's shape = (batch size, 1, output_size)
        cat_size = num_fields + deep_output_size
        self.fc = nn.Linear(cat_size, output_size)
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""feed forward of Deep Field-aware Factorization Machine Module
        
        Args:
            inputs (Dict[str, torch.Tensor]): Dictionary of inputs torch.Tensor
        
        Key-Values:
            first_order, shape = (batch size, num_fields, 1): first order outputs, i.e. outputs from nn.Embedding(vocab_size, 1)
            second_order, shape = (batch size, num_fields * num_fields, embed_size): field-aware second order outputs, :math:`x_{i, \text{field}_{j}}`
        
        Returns:
            torch.Tensor, shape = (batch size, output size), dtype = torch.float: outputs of Deep Field-aware Factorization Machine Module
        """
        # get batch size
        batch_size = inputs["first_order"].size(0)

        # first_order's shape = (batch size, number of fields, 1)
        # and the output's shape = (batch size, number of fields)
        first_out = inputs["first_order"]
        first_out = first_out.view(batch_size, -1)
        
        # second_order's shape = (batch size, number of fields * number of fields, embed size)
        # and the output's shape = (batch size, 1, output size)
        second_out = self.second_order(inputs["second_order"])
        second_out = second_out.view(batch_size, -1)

        # cat and feed-forward to nn.Linear
        outputs = torch.cat([second_out, first_out], dim=1)
        outputs = self.fc(outputs)

        return outputs
