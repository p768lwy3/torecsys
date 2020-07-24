import torch
import torch.nn as nn

from torecsys.utils.decorator import no_jit_experimental_by_namedtensor


class InnerProductNetworkLayer(nn.Module):
    r"""Layer class of Inner Product Network.
    
    Inner Product Network is an option in Product based Neural Network by Yanru Qu et at, 2016, 
    by calculating inner product between embedded tensors element-wisely to get cross features 
    interactions
    
    :Reference:

    #. `Yanru Qu et at, 2016. Product-based Neural Networks for User Response Prediction
    <https://arxiv.org/abs/1611.00144>`_.
    
    """

    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 num_fields: int):
        r"""Initialize InnerProductNetworkLayer
        
        Args:
            num_fields (int): Number of inputs' fields
        
        Attributes:
            row_idx (T), dtype = torch.long: 1st indices to index inputs in 2nd dimension for inner product.
            col_idx (T), dtype = torch.long: 2nd indices to index inputs in 2nd dimension for inner product.
        """
        # refer to parent class
        super(InnerProductNetworkLayer, self).__init__()

        # create row_idx and col_idx to index inputs for inner product
        row_idx = list()
        col_idx = list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row_idx.append(i)
                col_idx.append(j)
        self.row_idx = torch.LongTensor(row_idx)
        self.col_idx = torch.LongTensor(col_idx)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of InnerProductNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, NC2), dtype = torch.float: Output of InnerProductNetworkLayer
        """
        # calculate inner product between each field
        # inputs: emb_inputs, shape = (B, N, E)
        # output: inner, shape = (B, NC2, E)
        emb_inputs = emb_inputs.rename(None)
        inner = emb_inputs[:, self.row_idx] * emb_inputs[:, self.col_idx]
        inner.names = ("B", "N", "E")

        # aggregate on dimension E
        # inputs: inner, shape = (B, NC2, E)
        # output: outputs, shape = (B, NC2)
        outputs = torch.sum(inner, dim="E")

        # rename tensor names
        outputs.names = ("B", "O")

        return outputs
