import torch
import torch.nn as nn


class FieldAwareFactorizationMachineLayer(nn.Module):
    r"""FieldAwareFactorizationMachineLayer is a layer used in Field-Aware Factorization Machine 
    to calculate low dimension cross-features interaction per inputs field.
    
    :Reference:

    #. `Yuchin Juan et al, 2016. Field-aware Factorization Machines for CTR Prediction <https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf>`

    """
    def __init__(self, 
                 num_fields : int,
                 dropout_p  : float = 0.0):
        r"""initialize field-aware factorization machine layer module
        
        Args:
            num_fields (int): number of fields in inputs
            dropout_p (float, optional): dropout probability after field-aware factorization machine. Defaults to 0.0.
        """
        super(FieldAwareFactorizationMachineLayer, self).__init__()
        self.num_fields = num_fields
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed-forward calculation of field-aware factorization machine layer
        
        Notations:
            B: batch size
            N: number of fields
            E: embedding size

        Args:
            inputs (torch.Tensor), shape = (B, N * N, E), dtype = torch.float: features matrices of inputs
        
        Returns:
            torch.Tensor, shape = (B, 1, output size), dtype = torch.float: output of field-aware factorization machine layer
        """
        # chunk inputs' tensor into num_fields parts with shape = (B, N, E)
        inputs = torch.chunk(inputs, self.num_fields, dim=1)
        
        # calculate dot-product between efij and efji
        outputs = []
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                outputs.append(inputs[j][:, i] * inputs[i][:, j])
        
        # stack outputs into a tensor and pass into dropout layer
        outputs = torch.stack(outputs, dim=1)
        outputs = self.dropout(outputs)
        return outputs.unsqueeze(1)
