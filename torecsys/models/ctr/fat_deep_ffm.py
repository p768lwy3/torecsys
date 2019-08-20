from . import _CtrModel
from torecsys.layers import ComposeExcitationNetworkLayer, FieldAwareFactorizationMachineLayer, MultilayerPerceptronLayer
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import Callable, List


class FieldAttentiveDeepFieldAwareFactorizationMachineModel(_CtrModel):
    r"""FieldAttentiveDeepFieldAwareFactorizationMachineModel is a model of Field-Attentive Deep 
    Field-aware Factorization Machine, which is to apply a variant of SENet (an algorithm used in 
    computer vision originally) called CENet, to compose and excitation the field-aware embedding 
    vectors that used in field-aware factorization machine in the following way:
    
    #. Compose the field embedding matrices into a :math:`k * (n * n) * 1` matrices

    #. Excitation attentional weights with fully connect layers and apply the attentional weights on the 
    input fields embedding inputs

    #. apply field-aware factorization machine and deep neural network after compose excitation network

    :Reference:

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine <https://arxiv.org/abs/1905.06336>`_.
    
    """
    def __init__(self,
                 embed_size       : int,
                 num_fields       : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 reduction        : int, 
                 ffm_dropout_p    : float = 0.0,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        super(FieldAttentiveDeepFieldAwareFactorizationMachineModel, self).__init__()

        self.ffm = FieldAwareFactorizationMachineLayer(num_fields=num_fields, dropout_p=ffm_dropout_p)
        self.deep = MultilayerPerceptronLayer(
            output_size = deep_output_size,
            layer_sizes = deep_layer_sizes,
            embed_size  = embed_size,
            num_fields  = num_fields,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        )
        self.compose_excitation_network = ComposeExcitationNetworkLayer(num_fields, reduction)
    
    def forward(self, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""feed forward of FAT-DeepFFM model
        
        Args:
            field_emb_inputs (T), shape = (B, N * N, E), dtype = torch.float: field-aware embedding tensors, which is used in FFM and can be get by trs.inputs.base.FieldAwareIndexEmbedding()
        
        Returns:
            T, shape = (B, 1), dtype = torch.float: inference of FAT-DeepFFM model
        """
        # calculate attentional embedding matrices with compose excitation network,
        # where the output's shape = (B, N * N, E)
        aem = self.compose_excitation_network(field_emb_inputs)

        # sum the attentional embedding matrices into shape = (B, 1)
        first_order = aem.sum(dim=[1, 2]).unsqueeze(-1)

        # ffm part with inputs' shape = (B, N * N, E) and outputs' shape = (B, N, E)
        second_order = self.ffm(aem)

        # deep part with output's shape = (B, N, O) and sum into shape = (B, N, 1)
        second_order = self.deep(second_order).sum(dim=1)

        # sum the vectors as outputs
        outputs = first_order + second_order

        return outputs
