import torch
import torch.nn as nn


class PositionEmbeddingLayer(nn.Module):
    r"""Layer class of Position Embedding

    Position Embedding was used in Personalized Re-ranking Model :title:`Changhua Pei et al, 2019`[1],
    which is to add a trainable tensors per position to the session-based embedding features 
    tensor.

    :Reference:

    `Changhua Pei et al, 2019. Personalized Re-ranking for Recommendation <https://arxiv.org/abs/1904.06813>`_.

    """

    def __init__(self, max_num_position: int):
        r"""Initialize PositionEmbedding
        
        Args:
            max_num_position (int): Maximum number of position in a sequence.
        
        Attributes:
            bias (nn.Parameter): Bias variable of position in session.
        """
        # refer to parent class
        super(PositionEmbeddingLayer, self).__init__()

        # initialize bias variables
        self.bias = nn.Parameter(torch.Tensor(1, max_num_position, 1))

        # initialize bias variables with normalization
        nn.init.normal_(self.bias)

    def forward(self, session_embed_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of PositionEmbedding
        
        Args:
            session_embed_inputs (T), shape = (B, L, E), dtype = torch.float: Embedded feature tensors of session.
        
        Returns:
            T, shape = (B, L, E), dtype = torch.float: Output of PositionEmbedding
        """
        # add positional bias to session embedding features
        # inputs: session_embed_inputs, shape = (B, L, E)
        # inputs: self.bias, shape = (1, L, 1)
        # output: output, shape = (B, L, E)
        output = session_embed_inputs + self.bias

        return output
