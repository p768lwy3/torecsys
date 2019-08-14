import torch
import torch.nn as nn
from typing import Tuple


class AttentionalFactorizationMachineLayer(nn.Module):
    r"""AttentionalFactorizationMachineLayer is a layer used in Attentional Factorization Machine 
    to calculate low-dimension cross-features interactions by applying Attention-mechanism.
    
    :Reference:

    #. `Jun Xiao et al, 2017. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks∗ <https://arxiv.org/abs/1708.04617>`_.

    """
    def __init__(self, 
                 embed_size: int,
                 num_fields: int,
                 attn_size : int,
                 dropout_p : float = 0.1):
        r"""initialize attentional factorization machine layer module
        
        Args:
            embed_size (int): embedding size
            num_fields (int): number of fields in inputs
            attn_size (int): size of attention layer
            dropout_p (float, optional): dropout probability after attentional factorization machine. Defaults to 0.1.
        """
        # initialize nn.Module class
        super(AttentionalFactorizationMachineLayer, self).__init__()

        # to calculate attention score
        self.attn_score = nn.Sequential()
        self.attn_score.add_module("linear1", nn.Linear(embed_size, attn_size))
        self.attn_score.add_module("activation1", nn.ReLU())
        self.attn_score.add_module("out_proj", nn.Linear(attn_size, 1))
        self.attn_score.add_module("softmax1", nn.Softmax(dim=1))
        self.attn_score.add_module("dropout1", nn.Dropout(dropout_p))

        # to calculate inner-product
        self.row_idx = []
        self.col_idx = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row_idx.append(i)
                self.col_idx.append(j)
        
        # to dropout
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        r"""feed-forward calculation of attention factorization machine layer
        
        Notation:
            B: batch size
            N: number of fields
            E: embedding size

        Args:
            inputs (torch.Tensor), shape = (B, N, E), dtype = torch.float: features matrices of inputs
        
        Returns:
            Tuple[torch.Tensor], shape = ((B, 1, E) (B, NC2, 1)), dtype = torch.float: output and attention scores of Attentional Factorization Machine 
        """
        # calculate inner product between each field, hence inner's shape = (B, NC2, E)
        inner = inputs[:, self.row_idx] * inputs[:, self.col_idx]

        # calculate attention scores by inner product, hence scores' shape = (B, NC2, 1)
        attn_scores = self.attn_score(inner)
        
        # apply attention scores on inner-product and apply dropout before return
        outputs = torch.sum(attn_scores * inner, dim=1)
        outputs = self.dropout(outputs)
        return outputs.unsqueeze(1), attn_scores
    