import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Tuple

class AttentionalFactorizationMachineLayer(nn.Module):
    r"""Layer class of Attentional Factorization Machine (AFM). 
    
    Attentional Factorization Machine is to calculate interaction between each pair of features 
    by using element-wise product (i.e. Pairwise Interaction Layer), compressing interaction 
    tensors to a single representation. The output shape is (B, 1, E).
    
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
        # Refer to parent class
        super(AttentionalFactorizationMachineLayer, self).__init__()

        # Initialize sequential for attention
        self.attention = nn.Sequential()

        # Initialize module and add them to attention
        self.attention.add_module("Linear", nn.Linear(embed_size, attn_size))
        self.attention.add_module("Activation", nn.ReLU())
        self.attention.add_module("OutProj", nn.Linear(attn_size, 1))
        self.attention.add_module("Softmax", nn.Softmax(dim=1))
        self.attention.add_module("Dropout", nn.Dropout(dropout_p))

        # Generate rowidx and colidx to index inputs for inner product
        self.rowidx = list()
        self.colidx = list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.rowidx.append(i)
                self.colidx.append(j)
        self.rowidx = torch.LongTensor(self.rowidx)
        self.colidx = torch.LongTensor(self.colidx)
        
        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, emb_inputs: torch.Tensor) -> Tuple[torch.Tensor]:
        r"""Forward calculation of AttentionalFactorizationMachineLayer

        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            Tuple[T], shape = ((B, E) (B, NC2, 1)), dtype = torch.float: Output of AttentionalFactorizationMachineLayer and Attention weights.
        """
        # Calculate inner product
        # inputs: emb_inputs, shape = (B, N, E)
        # output: inner, shape = (B, NC2, E)
        emb_inputs = emb_inputs.rename(None)
        inner = emb_inputs[:, self.rowidx] * emb_inputs[:, self.colidx]
        inner.names = ("B", "N", "E")

        # Calculate attention scores
        # inputs: inner, shape = (B, NC2, E)
        # output: attn_scores, shape = (B, NC2, 1)
        attn_scores = self.attention(inner.rename(None))
        attn_scores.names = ("B", "N", "E")
        
        # Apply attention on inner product
        # inputs: inner, shape = (B, NC2, E)
        # inputs: attn_scores, shape = (B, NC2, 1)
        # output: outputs, shape = (B, E)
        outputs = (inner * attn_scores).sum(dim="N")

        # Apply dropout on outputs
        # inputs: outputs, shape = (B, E)
        # output: outputs, shape = (B, E)
        outputs = self.dropout(outputs)

        return outputs, attn_scores
    