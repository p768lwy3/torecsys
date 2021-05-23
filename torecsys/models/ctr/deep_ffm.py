from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import FFMLayer, DNNLayer
from torecsys.models.ctr import CtrBaseModel
from torecsys.utils.operations import combination


class DeepFieldAwareFactorizationMachineModel(CtrBaseModel):
    r"""
    Model class of Deep Field-aware Factorization Machine (Deep FFM).
    
    Deep Field-aware Factorization Machine was proposed by Yang et al in Tencent Social Ads competition 2017
    (this was called as Network on Field-aware Factorization Machine (NFFM), and was described and renamed to
    Deep Field-aware Factorization Machine in the research (Zhang et al, 2019).
    
    The model is a stack of Field Aware Factorization Machine and Deep Neural Network,

    #. First, calculate the interactions of features of second-order features (i.e. embedding matrices in FFM) by
    inner-product or hadamard product, Hence, let :math:`A` be the feature interaction vectors, :math:`A` will be
    calculate in the following formula: :math:`\text{Inner Product:} A = [v_{1, 2} \bigoplus v_{2, 1}, ..., v_{i,
    j} \bigoplus v_{j, i}, ..., v_{(n-1), n} \bigoplus v_{n, (n-1)}] :math:`\text{OR Hadamard Product:} A = [v_{1,
    2} \bigotimes v_{2, 1}, ..., v_{i, j} \bigotimes v_{j, i}, ..., v_{(n-1), n} \bigotimes v_{n, (n-1)}]

    #. Second, pass the matrices :math:`\text{A}` to a Deep Neural Network, where the 
    forward process is: :math:`\text{if i = 1,} x_{1} = \text{activation} ( W_{1} A + b_{1} )` 
    :math:`\text{otherwise,} x_{i} = \text{activation} ( W_{i} x_{i - 1} + b_{i})` 

    #. Finally, concatenate the above part and the linear part :math:`x_{linear}`, and pass 
    forward to a linear output layer: :math:`y(X) = W_{linear} x_{linear} + W_{second_order} x_{l} + b_{output}`

    :Reference:

    #. `Li Zhang et al, 2019. Field-aware Neural Factorization Machine for Click-Through Rate Prediction
    <https://arxiv.org/abs/1902.09096>`_.

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine
    <https://arxiv.org/abs/1905.06336>`_.

    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 deep_output_size: int,
                 deep_layer_sizes: List[int],
                 ffm_dropout_p: Optional[float] = None,
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize DeepFieldAwareFactorizationMachineModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            deep_output_size (int): output size of dense network
            deep_layer_sizes (List[int]): layer sizes of dense network
            ffm_dropout_p (float, optional): probability of Dropout in FFM. Defaults to 0.0
            deep_dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            deep_activation (torch.nn.Module, optional): activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.ffm = FFMLayer(num_fields=num_fields, dropout_p=ffm_dropout_p)

        inputs_size = combination(num_fields, 2)
        inputs_size *= embed_size
        self.deep = DNNLayer(
            inputs_size=inputs_size,
            output_size=deep_output_size,
            layer_sizes=deep_layer_sizes,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )

    def forward(self, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of DeepFieldAwareFactorizationMachineModel

        Args:
            field_emb_inputs (T), shape = (B, N * N, E), data_type = torch.float: Field aware embedded features tensors.
        
        Returns:
            T, shape = (B, O), data_type = torch.float: Output of DeepFieldAwareFactorizationMachineModel.
        """
        # Name the feat_inputs tensor for flatten
        field_emb_inputs.names = ('B', 'N', 'E',)

        # Get batch size from field_emb_inputs
        b = field_emb_inputs.size('B')

        # Aggregate field_emb_inputs on dimension N and E, and reshape it from (B, ) to (B, O = 1)
        # inputs: field_emb_inputs, shape = (B, N * N, E)
        # output: dffm_first, shape = (B, O = 1)
        dffm_first = field_emb_inputs.sum(dim=('N', 'E',)).unflatten('B', (('B', b,), ('O', 1,),))

        # Calculate with ffm layer forwardly
        # inputs: field_emb_inputs, shape = (B, N * N, E)
        # output: dffm_second, shape = (B, NC2, E)
        dffm_second = self.ffm(field_emb_inputs)

        # Flatten dffm_second
        # inputs: dffm_second, shape = (B, NC2, E)
        # output: dffm_second, shape = (B, E = NC2 * E)
        dffm_second = dffm_second.flatten(('N', 'E',), 'E')

        # Calculate with deep layer forwardly 
        # inputs: dffm_second, shape = (B, E = NC2 * E)
        # output: dffm_second, shape = (B, O = O_d)
        dffm_second = self.deep(dffm_second)

        # Aggregate dffm_second on dimension O,
        # inputs: dffm_second, shape = (B, O = O_d)
        # output: dffm_second, shape = (B, O = 1)
        dffm_second = dffm_second.sum('O', keepdim=True)

        # Add up dffm_second and dffm_first 
        # inputs: dffm_second, shape = (B, O = 1)
        # inputs: dffm_first, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = dffm_second + dffm_first

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
