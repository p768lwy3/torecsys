from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn


class InnerProductNetworkLayer(nn.Module):
    r"""Layer class of Inner Product Network, which is used in Product-based Neural Network
    :cite:`Yanru Qu et at, 2016`[1], by calculating inner product between embedded tensors 
    element-wisely to get cross features interactions.
    
    :Reference:

    #. `Yanru Qu et at, 2016. Product-based Neural Networks for User Response Prediction <https://arxiv.org/abs/1611.00144>`_.
    
    """
    @jit_experimental
    def __init__(self, 
                 num_fields: int):
        r"""Initialize InnerProductNetworkLayer
        
        Args:
            num_fields (int): Number of inputs' fields
        
        Attributes:
            rowidx (T), dtype = torch.long: 1st indices to index inputs in 2nd dimension for inner product.
            colidx (T), dtype = torch.long: 2nd indices to index inputs in 2nd dimension for inner product.
        """
        # refer to parent class
        super(InnerProductNetworkLayer, self).__init__()
        
        # create rowidx and colidx to index inputs for inner product
        rowidx = list()
        colidx = list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                rowidx.append(i)
                colidx.append(j)
        self.rowidx = torch.LongTensor(rowidx)
        self.colidx = torch.LongTensor(colidx)
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of InnerProductNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, 1, NC2), dtype = torch.float: Output of InnerProductNetworkLayer
        """
        # calculate inner product between each field,
        # inner's shape = (B, NC2, E)
        inner = emb_inputs[:, self.rowidx] * emb_inputs[:, self.colidx]

        # sum by third dimension, outputs' shape = (B, NC2, 1)
        outputs = torch.sum(inner, dim=2, keepdim=True)
        return outputs
