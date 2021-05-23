from typing import Tuple, Dict

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer


class BiasEncodingLayer(BaseLayer):
    """
    Layer class of Bias Encoding

    Bias Encoding was used in Deep Session Interest Network :title:`Yufei Feng et al, 2019`[1],
    which is to add three types of session-positional bias to session embedding tensors, 
    including: bias of session, bias of position in the session and bias of index in the 
    session.

    :Reference:

    #. `Yufei Feng, 2019. Deep Session Interest Network for Click-Through Rate Prediction
    <https://arxiv.org/abs/1905.06482>`_.
    
    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'session_embedding': ('B', 'L', 'E',),
            'session_index': ('B',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', 'L', 'E',)
        }

    def __init__(self,
                 embed_size: int,
                 max_num_session: int,
                 max_length: int):
        """
        Initialize BiasEncodingLayer
        
        Args:
            embed_size (int): size of embedding tensor
            max_num_session (int): maximum number of session in sequences.
            max_length (int): maximum number of position in sessions.
        """
        super().__init__()

        self.session_bias = nn.Parameter(torch.Tensor(max_num_session, 1, 1))
        self.position_bias = nn.Parameter(torch.Tensor(1, max_length, 1))
        self.item_bias = nn.Parameter(torch.Tensor(1, 1, embed_size))

        nn.init.normal_(self.session_bias)
        nn.init.normal_(self.position_bias)
        nn.init.normal_(self.item_bias)

    def forward(self, session_embed_inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Forward calculation of BiasEncodingLayer
        
        Args: session_embed_inputs ((T, T)), shape = ((B, L, E), (B, )), data_type = (torch.float, torch.long):
            embedded feature tensors of session and position of session in sequence.
        
        Returns:
            T, shape = (B, L, E), data_type = torch.float: output of BiasEncodingLayer
        """
        # reshape session_index and gather bias encoding from session_bias with session_index
        # inputs: session_embed_inputs[1], shape = (B, )
        # inputs: self.session_bias, shape = (S, 1, 1)
        # output: session_bias, shape = (B, 1, 1)
        session_index = session_embed_inputs[1]
        batch_size = session_index.size('B')
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
        output.names = ('B', 'L', 'E',)

        return output
