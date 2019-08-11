from . import _Inputs
import numpy as np
import torch
import torch.nn as nn
from Typing import List

class FieldAwareSingleIndexEmbedding(_Inputs):
    def __init__(self, embed_size: int, field_sizes: List[int]):
        super(FieldAwareSingleIndexEmbedding, self).__init__()
        self.num_fields = len(field_sizes)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_sizes), embed_size) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_sizes)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs : shape = (batch size, num fields), dtype = torch.long
        
        returns: torch.Tensor with (batch size, num_fields * num_fields, embedding size), 
        """
        inputs = inputs + inputs.new_tensor(self.offsets).unsqueeze(0)
        outputs = torch.cat([self.embeddings[i](inputs) for i in range(self.num_fields)], dim=1)
        return outputs
