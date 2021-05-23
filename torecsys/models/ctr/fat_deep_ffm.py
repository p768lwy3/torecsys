from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import CENLayer, FFMLayer, DNNLayer
from torecsys.models.ctr import CtrBaseModel
from torecsys.utils.operations import combination


class FieldAttentiveDeepFieldAwareFactorizationMachineModel(CtrBaseModel):
    """
    Model class of Field Attentive Deep Field Aware Factorization Machine (Fat DeepFFM).
    
    Field Attentive Deep Field Aware Factorization Machine is to apply CENet (a variant of SENet, an algorithm used in
    computer vision originally), to compose and excitation the field-aware embedding tensors that used in field-aware
    factorization machine in the following way:
    
    #. Compose the field embedding matrices into a :math:`k * (n * n) * 1` matrices.

    #. Excitation attentional weights with fully connect layers and apply the attentional weights 
    on the input fields embedding embedder.

    #. Apply field-aware factorization machine and deep neural network after compose excitation 
    network.

    :Reference:

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine
        <https://arxiv.org/abs/1905.06336>`_.
    
    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 deep_output_size: int,
                 deep_layer_sizes: List[int],
                 reduction: int,
                 ffm_dropout_p: Optional[float] = 0.0,
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize FieldAttentiveDeepFieldAwareFactorizationMachineModel
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of embedder' fields
            deep_output_size (int): output size of dense network
            deep_layer_sizes (List[int]): layer sizes of dense network
            reduction (int): reduction of CIN layer
            ffm_dropout_p (float, optional): probability of Dropout in FFM. Defaults to 0.0
            deep_dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            deep_activation (torch.nn.Module, optional): activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.cen = CENLayer(num_fields, reduction)
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
        Forward calculation of FieldAttentiveDeepFieldAwareFactorizationMachineModel
        
        Args:
            field_emb_inputs (T), shape = (B, N * N, E), data_type = torch.float: field aware embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: Output of FieldAttentiveDeepFieldAwareFactorizationMachineModel
        """
        # Name the feat_inputs tensor for getting batch size
        field_emb_inputs.names = ('B', 'N', 'E',)

        # Get batch size from field_emb_inputs
        b = field_emb_inputs.size('B')

        # Calculate attentional embedding matrices with compose excitation network,
        # where the output's shape = (B, N * N, E)
        aem = self.cen(field_emb_inputs.rename(None))
        aem.names = ('B', 'N', 'E',)

        # Sum the attentional embedding tensors into shape = (B, O = 1)
        first_order = aem.sum(dim=('N', 'E',)).unflatten('B', (('B', b,), ('O', 1,),))

        # ffm part with embedder' shape = (B, N * N, E) and outputs' shape = (B, N, E)
        second_order = self.ffm(aem)
        second_order.names = ('B', 'N', 'E',)
        second_order = second_order.flatten(('N', 'E',), 'E')

        # deep part with output's shape = (B, N, O) and sum into shape = (B, N, 1)
        second_order = self.deep(second_order)

        # Sum the vectors as outputs
        outputs = first_order + second_order

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
