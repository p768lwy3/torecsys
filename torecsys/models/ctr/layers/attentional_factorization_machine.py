import torch
import torch.nn as nn
from Tyipng import Tuple

class AttentionalFactorizationMachineLayer(nn.Module):
    def __init__(self, 
                 embed_size: int,
                 num_fields: int,
                 attn_size : int,
                 dropout_p : float):
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
        self.out_dropout = nn.Dropout(dropout_p)
    
    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        """
        shape = (b, n, e)
        
        output shape = (b, embed),
        and shape = (b, numfield C 2, 1)
        """
        inner = inputs[:, self.row_idx] * inputs[:, self.col_idx] # (b, nC2, embed)
        attn_scores = self.attn_score(inner) # (b, nc2, 1)
        outputs = torch.sum(attn_scores * inner, dim=1)
        outputs = self.out_dropout(outputs)
        return outputs, attn_scores
    