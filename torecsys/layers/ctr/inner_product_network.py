import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor

class InnerProductNetworkLayer(nn.Module):
    r"""Layer class of Inner Product Network.
    
    Inner Product Network is in Product based Neural Network :title:`Yanru Qu et at, 2016`[1], 
    by calculating inner product between embedded tensors element-wisely to get cross features 
    interactions
    
    :Reference:

    #. `Yanru Qu et at, 2016. Product-based Neural Networks for User Response Prediction <https://arxiv.org/abs/1611.00144>`_.
    
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 num_fields: int):
        r"""Initialize InnerProductNetworkLayer
        
        Args:
            num_fields (int): Number of inputs' fields
        
        Attributes:
            rowidx (T), dtype = torch.long: 1st indices to index inputs in 2nd dimension for inner product.
            colidx (T), dtype = torch.long: 2nd indices to index inputs in 2nd dimension for inner product.
        """
        # Refer to parent class
        super(InnerProductNetworkLayer, self).__init__()
        
        # Create rowidx and colidx to index inputs for inner product
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
            T, shape = (B, NC2), dtype = torch.float: Output of InnerProductNetworkLayer
        """
        # Calculate inner product between each field
        # inputs: emb_inputs, shape = (B, N, E)
        # output: inner, shape = (B, NC2, E)
        emb_inputs = emb_inputs.rename(None)
        inner = emb_inputs[:, self.rowidx] * emb_inputs[:, self.colidx]
        inner.names = ("B", "N", "E")

        # Aggregate on dimension E
        # inputs: inner, shape = (B, NC2, E)
        # output: outputs, shape = (B, NC2)
        outputs = torch.sum(inner, dim="E")

        # Rename tensor names
        outputs.names = ("B", "O")
        
        return outputs
