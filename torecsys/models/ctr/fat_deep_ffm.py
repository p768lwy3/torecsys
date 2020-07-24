from typing import Callable, List

import torch
import torch.nn as nn

from torecsys.layers import CENLayer, FFMLayer, DNNLayer
from torecsys.utils.decorator import no_jit_experimental_by_namedtensor
from torecsys.utils.operations import combination
from . import _CtrModel


class FieldAttentiveDeepFieldAwareFactorizationMachineModel(_CtrModel):
    r"""Model class of Field Attentive Deep Field Aware Factorization Machine (Fat DeepFFM).
    
    Field Attentive Deep Field Aware Factorization Machine is to apply CENet (a variant of SENet, 
    an algorithm used in computer vision originally), to compose and excitation the field-aware 
    embedding tensors that used in field-aware factorization machine in the following way:
    
    #. Compose the field embedding matrices into a :math:`k * (n * n) * 1` matrices.

    #. Excitation attentional weights with fully connect layers and apply the attentional weights 
    on the input fields embedding inputs.

    #. Apply field-aware factorization machine and deep neural network after compose excitation 
    network.

    :Reference:

    #. `Junlin Zhang et al, 2019. FAT-DeepFFM: Field Attentive Deep Field-aware Factorization Machine <https://arxiv.org/abs/1905.06336>`_.
    
    """

    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 deep_output_size: int,
                 deep_layer_sizes: List[int],
                 reduction: int,
                 ffm_dropout_p: float = 0.0,
                 deep_dropout_p: List[float] = None,
                 deep_activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        """Initialize FieldAttentiveDeepFieldAwareFactorizationMachineModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            deep_output_size (int): Output size of dense network
            deep_layer_sizes (List[int]): Layer sizes of dense network
            reduction (int): Reduction of CIN layer
            ffm_dropout_p (float, optional): Probability of Dropout in FFM. 
                Defaults to 0.0.
            deep_dropout_p (List[float], optional): Probability of Dropout in dense network. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of dense network. 
                Defaults to nn.ReLU().
        
        Attributes:
            cen (nn.Module): Module of compose excitation network layer.
            ffm (nn.Module): Module of field-aware factorization machine layer.
            deep (nn.Module): Module of dense layer.
        """
        # refer to parent class
        super(FieldAttentiveDeepFieldAwareFactorizationMachineModel, self).__init__()

        # initialize compose excitation network
        self.cen = CENLayer(num_fields, reduction)

        # initialize ffm layer
        self.ffm = FFMLayer(num_fields=num_fields, dropout_p=ffm_dropout_p)

        # calculate the output's size of ffm, i.e. inputs' size of DNNLayer
        inputs_size = combination(num_fields, 2)
        inputs_size *= embed_size

        # initialize dense layer
        self.deep = DNNLayer(
            inputs_size=inputs_size,
            output_size=deep_output_size,
            layer_sizes=deep_layer_sizes,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )

    def forward(self, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of FieldAttentiveDeepFieldAwareFactorizationMachineModel
        
        Args:
            field_emb_inputs (T), shape = (B, N * N, E), dtype = torch.float: Field aware embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of FieldAttentiveDeepFieldAwareFactorizationMachineModel.
        """
        # get batch size from field_emb_inputs
        b = field_emb_inputs.size("B")

        # calculate attentional embedding matrices with compose excitation network,
        # where the output's shape = (B, N * N, E)
        aem = self.cen(field_emb_inputs.rename(None))
        aem.names = ("B", "N", "E")

        # sum the attentional embedding tensors into shape = (B, O = 1)
        first_order = aem.sum(dim=["N", "E"]).unflatten("B", [("B", b), ("O", 1)])

        # ffm part with inputs' shape = (B, N * N, E) and outputs' shape = (B, N, E)
        second_order = self.ffm(aem)
        second_order.names = ("B", "N", "E")
        second_order = second_order.flatten(["N", "E"], "E")

        # deep part with output's shape = (B, N, O) and sum into shape = (B, N, 1)
        second_order = self.deep(second_order)

        # sum the vectors as outputs
        outputs = first_order + second_order

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return outputs
