import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Tuple


class AttentionalFactorizationMachineLayer(nn.Module):
    r"""Layer class of Attentional Factorization Machine (AFM) to calculate interaction between each 
    pair of features by using element-wise product (i.e. Pairwise Interaction Layer), compressing 
    interaction tensors to a single representation. The output shape is (B, 1, E).
    
    :Reference:

    #. `Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks∗ <https://arxiv.org/abs/1708.04617>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 embed_size: int,
                 num_fields: int,
                 attn_size : int,
                 dropout_p : float = 0.1):
        r"""Initialize AttentionalFactorizationMachineLayer
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            attn_size (int): Size of attention layer
            dropout_p (float, optional): Probability of Dropout in AFM. 
                Defaults to 0.1.
        
        Attributes:
            attention (torch.nn.Sequential): Sequential of Attention-layers.
            rowidx (T), dtype = torch.long: 1st indices to index inputs in 2nd dimension for inner product.
            colidx (T), dtype = torch.long: 2nd indices to index inputs in 2nd dimension for inner product.
            dropout (torch.nn.Module): Dropout layer.
        """
        # refer to parent class
        super(AttentionalFactorizationMachineLayer, self).__init__()

        # initialize sequential for Attention
        self.attention = nn.Sequential()

        # add modules to sequential of Attention
        self.attention.add_module("linear", nn.Linear(embed_size, attn_size))
        self.attention.add_module("activation", nn.ReLU())
        self.attention.add_module("out_proj", nn.Linear(attn_size, 1))
        self.attention.add_module("softmax", nn.Softmax(dim=1))
        self.attention.add_module("dropout", nn.Dropout(dropout_p))

        # create rowidx and colidx to index inputs for inner product
        self.rowidx = list()
        self.colidx = list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.rowidx.append(i)
                self.colidx.append(j)
        
        self.rowidx = torch.LongTensor(self.rowidx)
        ## self.rowidx.names = ("I", )
        self.colidx = torch.LongTensor(self.colidx)
        ## self.colidx.names = ("I", )
        
        # initialize dropout layer before return
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, emb_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        r"""Forward calculation of AttentionalFactorizationMachineLayer

        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            Tuple[T], shape = ((B, E) (B, NC2, 1)), dtype = torch.float: Output of AttentionalFactorizationMachineLayer and Attention weights.
        """
        # calculate inner product between each field,
        # inner's shape = (B, NC2, E)
        ## inner = emb_inputs[:, self.rowidx] * emb_inputs[:, self.colidx]
        emb_inputs = emb_inputs.rename(None)
        inner = emb_inputs[:, self.rowidx] * emb_inputs[:, self.colidx]
        inner.names = ("B", "N", "E")

        # calculate attention scores by inner product,
        # scores' shape = (B, NC2, 1)
        attn_scores = self.attention(inner.rename(None))
        attn_scores.names = ("B", "N", "E")
        
        # apply attention scores on inner-product
        ## outputs = torch.sum(attn_scores * inner, dim=1)
        outputs = (attn_scores * inner).sum(dim="N")

        # apply dropout before return
        outputs = self.dropout(outputs)

        return outputs, attn_scores
    