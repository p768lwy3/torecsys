from . import _CtrModel
from torecsys.layers import FFMLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental
from torecsys.utils import combination
import torch
import torch.nn as nn
from typing import Callable, List


class DeepFieldAwareFactorizationMachineModel(_CtrModel):
    r"""DeepFieldAwareFactorizationMachineModel is a model of Deep Field-aware Factorization 
    Machine (DeepFFM) proposed by Yang et al in Tencent Social Ads competition 2017 (they called
    this model Network on Field-aware Factorization Machine - NFFM), and described and rename to
    Deep Field-aware Factorization Machine by Zhang et al in their research (Zhang et al, 2019). 
    The model is a stack of Field Aware Factorization Machine and Deep Neural Network, 

    #. First, calculate the interactions of features of second-order features (i.e. embedding \
    matrices in FFM) by inner-product or hadamard product, Hence, let :math:`A` be the feature
    interaction vectors, :math:`A` will be calculate in the following formula: 
    :math:`\text{Inner Product:} A = [v_{1, 2} \bigoplus v_{2, 1}, ..., v_{i, j} \bigoplus v_{j, i}, ..., v_{(n-1), n} \bigoplus v_{n, (n-1)}]
    :math:`\text{OR Hadamard Product:} A = [v_{1, 2} \bigotimes v_{2, 1}, ..., v_{i, j} \bigotimes v_{j, i}, ..., v_{(n-1), n} \bigotimes v_{n, (n-1)}]

    #. Second, pass the matrices :math:`\text{A}` to a Deep Neural Network, where the 
    forward process is: :math:`\text{if i = 1,} x_{1} = \text{activation} ( W_{1} A + b_{1} )` 
    :math:`\text{otherwise,} x_{i} = \text{activation} ( W_{i} x_{i - 1} + b_{i})` 

    #. Finally, concatenate the above part and the linear part :math:`x_{linear}, and pass forward to a linear 
    output layer:
    :math:`y(X) = W_{linear} x_{linear} + W_{second_order} x_{l} + b_{output}` .

    FieldAwareNeuralFactorizationMachineModel is a model of Field-aware Neural
    Factorization Machine, which is a stack of Field Aware Factorization Machine and 
    Deep Neural Network, with the following calculation:
    First, calculate bi-interaction of features by field aware factorization machine: 
    :math:`y_{FFM} = \sum_{i=1}^{N} \sum_{j=i+1}^{N} x_{i} v_{i} \bigotimes  x_{j} v_{j}` .
    Then feed the interactions' representation to deep neural network: :math:`y_{i} = \text{Activation} ( w_{i} y_{i - 1} + b_{i} )` , 
    where :math:`y_{0} = y_{FM}` for the inputs of the first layer in deep neural network.

    :Reference:

    #. `Li Zhang et al, 2019. Field-aware Neural Factorization Machine for Click-Through Rate Prediction <https://arxiv.org/abs/1902.09096>`_.

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine <https://arxiv.org/abs/1905.06336>`_.

    """
    @jit_experimental
    def __init__(self,
                 embed_size       : int,
                 num_fields       : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 ffm_dropout_p    : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU(),
                 output_size      : int = 1):
        r"""initialize Deep Field-aware Factorization Machine Model
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            deep_output_size (int): output size of deep neural network
            deep_layer_sizes (List[int]): layer sizes of deep neural network
            ffm_dropout_p (float, optional): dropout probability after ffm layer. Defaults to 0.0.
            deep_dropout_p (List[float], optional): dropout probability after each deep neural network. Allow: [None, List[float]]. Defaults to None.
            deep_activation (Callable[[T], T], optional): activation after each deep neural network. Allow: [None, Callable[[T], T]]. Defaults to nn.ReLU().
            output_size (int, optional): output size of linear transformation after concatenate. Defaults to 1.
        """
        # initialize nn.Module class
        super(DeepFieldAwareFactorizationMachineModel, self).__init__()

        # sequential of second-order part in inputs
        self.second_order = nn.Sequential()
        # ffm's input shape = (B, N * N, E)
        # ffm's output shape = (B, NC2, E)
        self.second_order.add_module("ffm", FFMLayer(
            num_fields=num_fields, 
            dropout_p=ffm_dropout_p
        ))

        # calculate the output's size of ffm, i.e. inputs' size of DNNLayer
        inputs_size = combination(num_fields, 2)

        # deep's input shape = (B, NC2, E)
        # deep's output shape = (B, 1, O)
        self.second_order.add_module("deep", DNNLayer(
            output_size=deep_output_size, 
            layer_sizes=deep_layer_sizes, 
            embed_size=embed_size, 
            num_fields=inputs_size, 
            dropout_p=deep_dropout_p, 
            activation=deep_activation
        ))

    
    def forward(self, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of Deep Field-aware Factorization Machine Model

        Args:
            field_emb_inputs (T), shape = (B, N * N, E): field-aware second order outputs, :math:`x_{i, \text{field}_{j}}`
        
        Returns:
            torch.Tensor, shape = (B, 1), dtype = torch.float: outputs of Deep Field-aware Factorization Machine Model
        """

        # feat_inputs's shape = (B, N * N, E)
        # and the output's shape = (B, 1)
        dffm_first = field_emb_inputs.sum(dim=[1, 2]).unsqueeze(-1)
        
        # field_emb_inputs's shape = (B, N * N, E)
        # and the output's shape = (B, 1)
        dffm_second = self.second_order(field_emb_inputs)
        dffm_second = dffm_second.sum(dim=1)

        # cat and feed-forward to nn.Linear
        outputs = dffm_first + dffm_second

        return outputs
