from . import _CtrModel
from torecsys.layers import FFMLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn


class FieldAwareFactorizationMachineModel(_CtrModel):
    r"""Model class of Field-aware Factorization Machine (FFM).
    
    Field-aware Factorization Machine is a model to calculate the interaction of features for 
    each field with different embedding vectors, instead of a universal vectors.

    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size    : int,
                 num_fields    : int,
                 dropout_p     : float = 0.0):
        r"""Initialize FieldAwareFactorizationMachineModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            dropout_p (float, optional): Probability of Dropout in FFM. 
                Defaults to 0.0.
        
        Attributes:
            ffm (nn.Module): Module of field-aware factorization machine layer.
            bias (nn.Parameter): Parameter of bias of field-aware factorization machine.
        """
        # refer to parent class
        super(FieldAwareFactorizationMachineModel, self).__init__()

        # initialize ffm layer
        self.ffm = FFMLayer(num_fields, dropout_p=dropout_p)
            
        # initialize bias parameter
        self.bias = nn.Parameter(torch.zeros((1, 1), names=("B", "O")))
        nn.init.uniform_(self.bias.data)

    def forward(self, feat_inputs: torch.Tensor, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of FieldAwareFactorizationMachineModel
        
        Args:
            feat_inputs (T), shape = (B, N, 1), dtype = torch.float: Linear Features tensors.
            field_emb_inputs (T), shape = (B, N * N, E), dtype = torch.float: Field aware embedded features tensors.
        
        Returns:
            torch.Tensor, shape = (B, O), dtype = torch.float: Output of FieldAwareFactorizationMachineModel.
        """
        # get batch size from field_emb_inputs
        b = field_emb_inputs.size("B")

        # aggregate feat_inputs on dimension N and rename dimension E to O
        # hence, ffm_first's shape = (B, O = 1)
        # inputs: feat_inputs, shape = (B, N, E = 1)
        # output: ffm_first, shape = (B, O = 1)
        ffm_first = feat_inputs.sum(dim="N").rename(E="O")

        # inputs: field_emb_inputs, shape = (B, N * N, E) 
        # output: ffm_second, shape = (B, N, E)
        ffm_second = self.ffm(field_emb_inputs)

        # aggregate ffm_second on dimension [N, E], 
        # then reshape the sum from (B, ) to (B, O = 1)
        # inputs: ffm_second, shape = (B, N, E)
        # output: ffm_second, shape = (B, O = 1)
        ## ffm_second = ffm_second.sum(dim=[1, 2]).unsqueeze(-1)
        ffm_second = ffm_second.sum(dim=["N", "E"]).unflatten("B", [("B", b), ("O", 1)])

        # add up bias, fm_first and ffm_sceond
        # inputs: ffm_second, shape = (B, O = 1)
        # inputs: ffm_first, shape = (B, O = 1)
        # inputs: bias, shape = (B, O = 1)
        # output: outputs, shape = (B, O = 1)
        outputs = ffm_second + ffm_first + self.bias

        # since autograd does not support Named Tensor at this stage,
        # drop the name of output tensor.
        outputs = outputs.rename(None)

        return outputs
