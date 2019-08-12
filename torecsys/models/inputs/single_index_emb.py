from . import _Inputs
import torch
import torch.nn as nn


class SingleIndexEmbedding(_Inputs):
    r"""SingleIndexEmbedding is a embedding field to pass a single index and lookup the dense embedding vector of it
    """
    def __init__(self,
                 embed_size   : int,
                 field_size   : int,
                 padding_idx  : int = None,
                 nn_embedding : nn.Parameter = None,
                 **kwargs):
        r"""initialize the single index embedding
        
        Args:
            embed_size (int): embedding size
            field_size (int): field sizes
            padding_idx (int, optional): padding index of field. Defaults to None.
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None.
        """
        super(SingleIndexEmbedding, self).__init__()
        if nn_embedding is not None:
            self.embed_size = nn_embedding.size(1)
            self.embedding = torch.jit.trace(
                nn.Embedding.from_pretrained(nn_embedding),
                (torch.randint(low=0, high=self.field_size, size=(1, )).long())
            )
        else:
            self.embed_size = embed_size
            self.embedding = torch.jit.trace(
                nn.Embedding(field_size, embed_size, padding_idx=padding_idx, **kwargs),
                (torch.randint(low=0, high=self.field_size, size=(1, )).long())
            )
        self.length = self.embed_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Return embedding vectors of inputs
        
        Args:
            inputs (torch.Tensor), shape = (batch size, num fields), dtype = torch.long: torch.Tensor of inputs, \
                    where they are the indices of fields (i.e. length of column of torch.Tensor = number of fields) for each row
        
        Returns:
            torch.Tensor, (batch size, num_fields, embedding size): embedding vectors :math:`\bm{E} = \bm{\Big[} e_{\text{index}_{i}} \footnotesize{\text{, for} \ i = \text{i-th field}} \bm{\Big]}`
        """
        return self.embedding(inputs)
