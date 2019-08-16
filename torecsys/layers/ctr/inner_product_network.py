from torecsys.utils.logging.decorator import jit_experimental
import torch
import torch.nn as nn


class InnerProductNetworkLayer(nn.Module):
    r"""InnerProductNetworkLayer is a layer used in Product-based Neural Network to calculate 
    element-wise cross-feature interactions by inner-product of matrix multiplication.
    
    :Reference:

    #. `Yanru Qu et at, 2016. Product-based Neural Networks for User Response Prediction <https://arxiv.org/abs/1611.00144>`_.
    
    """
    @jit_experimental
    def __init__(self, 
                 num_fields: int):
        r"""initialize inner product network layer module
        
        Args:
            num_fields (int): number of fields in inputs
        """
        super(InnerProductNetworkLayer, self).__init__()
        
        # indices for inner product
        rowidx = list()
        colidx = list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                rowidx.append(i)
                colidx.append(j)
        self.rowidx = torch.LongTensor(rowidx)
        self.colidx = torch.LongTensor(colidx)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed-forward calculation of inner product network
        
        Args:
            inputs (torch.Tensor), shape = (B, N, E), dtype = torch.float: features vectors of inputs
        
        Returns:
            torch.Tensor, shape = (B, 1, NC2), dtype = torch.float: output of inner product network
        """
        outputs = torch.sum(inputs[:, self.rowidx] * inputs[:, self.colidx], dim=2)
        return outputs.unsqueeze(1)
