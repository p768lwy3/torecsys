from typing import Tuple

import torch
import torch.nn as nn

from torecsys.utils.decorator import no_jit_experimental


class BiasEncodingLayer(nn.Module):
    r"""Layer class of Bias Encoding 

    Bias Encoding was used in Deep Session Interest Network :title:`Yufei Feng et al, 2019`[1],
    which is to add three types of session-positional bias to session embedding tensors, 
    including: bias of session, bias of position in the session and bias of index in the 
    session.

    :Reference:

    #. `Yufei Feng, 2019. Deep Session Interest Network for Click-Through Rate Prediction
    <https://arxiv.org/abs/1905.06482>`_.
    
    """

    @no_jit_experimental
    def __init__(self,
                 embed_size: int,
                 max_num_session: int,
                 max_num_position: int):
        r"""Initialize BiasEncodingLayer
        
        Args:
            embed_size (int): Size of embedding tensor
            max_num_session (int): Maximum number of session in sequences.
            max_num_position (int): Maximum number of position in sessions.
        
        Attributes:
            session_bias (nn.Parameter): Bias variable of session in sequence.
            position_bias (nn.Parameter): Bias variable of position in session.
            item_bias (nn.Parameter): Bias variable of embedding features in item.
        """
        # refer to parent class
        super(BiasEncodingLayer, self).__init__()

        # initialize bias encoding variables
        self.session_bias = nn.Parameter(torch.Tensor(max_num_session, 1, 1))
        self.position_bias = nn.Parameter(torch.Tensor(1, max_num_position, 1))
        self.item_bias = nn.Parameter(torch.Tensor(1, 1, embed_size))

        # initialize bias variables with normalization
        nn.init.normal_(self.session_bias)
        nn.init.normal_(self.position_bias)
        nn.init.normal_(self.item_bias)

    def forward(self, session_embed_inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        r"""Forward calculation of BiasEncodingLayer
        
        Args: session_embed_inputs ((T, T)), shape = ((B, L, E), (B, )), dtype = (torch.float, torch.long): Embedded
        feature tensors of session and Position of session in sequence.
        
        Returns:
            T, shape = (B, L, E), dtype = torch.float: Output of BiasEncodingLayer
        """
        # reshape session_index and gather bias encoding from session_bias with session_index
        # inputs: session_embed_inputs[1], shape = (B, )
        # inputs: self.session_bias, shape = (S, 1, 1)
        # output: session_bias, shape = (B, 1, 1)
        session_index = session_embed_inputs[1]
        batch_size = session_index.size("B")
        session_index = session_index.rename(None)
        session_index = session_index.view(batch_size, 1, 1)
        session_bias = self.session_bias.gather(dim=0, index=session_index)

        # add encoding bias to session_embed_inputs
        # inputs: session_embed_inputs[0], shape = (B, L, E)
        # inputs: session_bias, shape = (B, 1, 1)
        # inputs: self.position_bias, shape = (1, L, 1)
        # inputs: self.item_bias, shape = (1, 1, E)
        # output: output, shape = (B, L, E)
        output = session_embed_inputs[0] + session_bias + self.position_bias + self.item_bias

        return output
