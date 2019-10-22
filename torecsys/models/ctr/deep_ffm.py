from . import _CtrModel
from torecsys.layers import FFMLayer, DNNLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from torecsys.utils.utils import combination
import torch
import torch.nn as nn
from typing import Callable, List


class DeepFieldAwareFactorizationMachineModel(_CtrModel):
    r"""Model class of Deep Field-aware Factorization Machine (Deep FFM).
    
    Deep Field-aware Factorization Machine was proposed by Yang et al in Tencent 
    Social Ads competition 2017 (this was called as Network on Field-aware Factorization 
    Machine (NFFM), and was described and renamed to Deep Field-aware Factorization Machine 
    in the research (Zhang et al, 2019). 
    
    The model is a stack of Field Aware Factorization Machine and Deep Neural Network, 

    #. First, calculate the interactions of features of second-order features (i.e. embedding 
    matrices in FFM) by inner-product or hadamard product, Hence, let :math:`A` be the feature
    interaction vectors, :math:`A` will be calculate in the following formula: 
    :math:`\text{Inner Product:} A = [v_{1, 2} \bigoplus v_{2, 1}, ..., v_{i, j} \bigoplus v_{j, i}, ..., v_{(n-1), n} \bigoplus v_{n, (n-1)}]
    :math:`\text{OR Hadamard Product:} A = [v_{1, 2} \bigotimes v_{2, 1}, ..., v_{i, j} \bigotimes v_{j, i}, ..., v_{(n-1), n} \bigotimes v_{n, (n-1)}]

    #. Second, pass the matrices :math:`\text{A}` to a Deep Neural Network, where the 
    forward process is: :math:`\text{if i = 1,} x_{1} = \text{activation} ( W_{1} A + b_{1} )` 
    :math:`\text{otherwise,} x_{i} = \text{activation} ( W_{i} x_{i - 1} + b_{i})` 

    #. Finally, concatenate the above part and the linear part :math:`x_{linear}, and pass 
    forward to a linear output layer:
    :math:`y(X) = W_{linear} x_{linear} + W_{second_order} x_{l} + b_{output}`_.

    :Reference:

    #. `Li Zhang et al, 2019. Field-aware Neural Factorization Machine for Click-Through Rate Prediction <https://arxiv.org/abs/1902.09096>`_.

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine <https://arxiv.org/abs/1905.06336>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size       : int,
                 num_fields       : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 output_size      : int = 1,
                 ffm_dropout_p    : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize DeepFieldAwareFactorizationMachineModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            deep_output_size (int): Output size of dense network
            deep_layer_sizes (List[int]): Layer sizes of dense network
            output_size (int, optional): Output size of model, 
                i.e. output size of the projection layer. 
                Defaults to 1.
            ffm_dropout_p (float, optional): Probability of Dropout in FFM. 
                Defaults to 0.0.
            deep_dropout_p (List[float], optional): Probability of Dropout in dense network. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of dense network. 
                Defaults to nn.ReLU().
        
        Attributes:
            ffm (nn.Module): Module of field-aware factorization machine layer.
            deep (nn.Module): Module of dense layer.
        """
        # refer to parent class
        super(DeepFieldAwareFactorizationMachineModel, self).__init__()

        # initialize ffm layer with the input's shape = (B, N * N, E) 
        # and output shape = (B, NC2, E)
        self.ffm = FFMLayer(
            num_fields = num_fields, 
            dropout_p  = ffm_dropout_p
        )

        # calculate the output's size of ffm, i.e. inputs' size of DNNLayer
        inputs_size = combination(num_fields, 2)
        inputs_size *= embed_size

        # initialize dense layer with the input's shape = (B, NC2, E)
        # and output shape = (B, O)
        self.deep = DNNLayer(
            inputs_size = inputs_size,
            output_size = deep_output_size, 
            layer_sizes = deep_layer_sizes, 
            dropout_p   = deep_dropout_p, 
            activation  = deep_activation
        )

    def forward(self, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of DeepFieldAwareFactorizationMachineModel

        Args:
            field_emb_inputs (T), shape = (B, N * N, E), dtype = torch.float: Field aware embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of DeepFieldAwareFactorizationMachineModel.
        """
        # feat_inputs's shape = (B, N * N, E)
        # and the output's shape = (B, O = 1)
        ## dffm_first = field_emb_inputs.sum(dim=[1, 2]).unsqueeze(-1)
        b = field_emb_inputs.size("B")
        dffm_first = field_emb_inputs.sum(dim=["N", "E"]).unflatten("B", [("B", b), ("O", 1)])
        
        # field_emb_inputs's shape = (B, N * N, E)
        # and the output's shape = (B, 1)
        ## dffm_second = self.second_order(field_emb_inputs)
        ## dffm_second = dffm_second.sum(dim=1)

        # calculate ffm output, where the output's shape = (B, NC2, E)
        dffm_second = self.ffm(field_emb_inputs)
        # flatten the output to shape = (B, NC2 * E)
        dffm_second = dffm_second.flatten(["N", "E"], "E")
        # calculate deep output, where the output's shape = (B, O_d)
        dffm_second = self.deep(dffm_second)
        # sum on dimesnion of O, where the output's shape = (B, O = 1)
        dffm_second = dffm_second.sum("O", keepdim=True)

        # sum dffm_first and dffm_second and the output's shape = (B, O = 1)
        outputs = dffm_first + dffm_second

        # since autograd does not support Named Tensor at this stage,
        # drop the name of output tensor.
        outputs = outputs.rename(None)
        
        return outputs
