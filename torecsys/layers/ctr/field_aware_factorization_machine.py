from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn


class FieldAwareFactorizationMachineLayer(nn.Module):
    r"""Layer class of Field aware Factorization Machine (FFM) :title:`Yuchin Juan et al, 2016`[1],  
    to calculate element-wise cross features interaction per fields for sparse field by using dot 
    product between field-wise feature tensors
    
    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 num_fields : int,
                 dropout_p  : float = 0.0):
        r"""Initialize FieldAwareFactorizationMachineLayer

        Args:
            num_fields (int): Number of inputs' fields
            dropout_p (float, optional): Probability of Dropout in FFM. 
                Defaults to 0.0.
        
        Attributes:
            num_fields (int): Number of inputs' fields.
            dropout (torch.nn.Module): Dropout layer.
        """
        # refer to parent class
        super(FieldAwareFactorizationMachineLayer, self).__init__()

        # bind num_fields to num_fields
        self.num_fields = num_fields

        # initialize dropout layer before return
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, field_emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of FieldAwareFactorizationMachineLayer

        Args:
            field_emb_inputs (T), shape = (B, N * N, E), dtype = torch.float: Field aware embedded features tensors.
        
        Returns:
            T, shape = (B, NC2, E), dtype = torch.float: Output of FieldAwareFactorizationMachineLayer
        """
        # initialize list to store tensors temporarily for output 
        outputs = list()

        # chunk inputs' tensor into num_fields parts with shape = (B, N, E)
        field_emb_inputs = field_emb_inputs.rename(None)
        field_emb_inputs = torch.chunk(field_emb_inputs, self.num_fields, dim=1)
        
        # calculate dot-product between e_{i, fj} and e_{j, fi}
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                outputs.append(field_emb_inputs[j][:, i] * field_emb_inputs[i][:, j])
        
        # stack outputs into a tensor and pass into dropout layer
        outputs = torch.stack(outputs, dim=1)

        # apply dropout before return
        outputs = self.dropout(outputs)

        # set names to the outputs' tensor
        outputs.names = ("B", "N", "E")
        
        return outputs
