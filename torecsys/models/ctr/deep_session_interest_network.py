from . import _CtrModel
import torch
import torch.nn as nn
from torecsys.layers import BiasEncodingLayer
from torecsys.utils.decorator import jit_experimental
from typing import Callable, List, Tuple

class DeepSessionInterestNetworkModel(_CtrModel):
    r"""Model class of Deep Session Interest Network (DSIN), which is a stack of self attention 
    and bi-lstm to extract features' from sessions and calculate interaction between features 
    before dense layer.

    :Reference:

    #. `Yufei Feng, 2019. Deep Session Interest Network for Click-Through Rate Prediction <https://arxiv.org/abs/1905.06482>`_.

    """
    def __init__(self,
                 embed_size                       : int,
                 max_num_session                  : int,
                 max_num_position                 : int,
                 interest_extractor_num_heads     : int,
                 interest_interacting_hidden_size : int,
                 interest_extractor_dropout       : float = 0.0,
                 interest_extractor_bias          : bool = True,
                 interest_interacting_num_layers  : int = 1,
                 interest_interacting_dropout     : float = 0.0,
                 interest_interacting_bias        : bool = True,
                 use_bias_encoding                : bool = True):
        r"""Initialize DeepSessionInterestNetworkModel
        
        Args:
            embed_size (int): Size of embedding tensor
            max_num_session (int): Maximum number of session in sequences.
            max_num_position (int): Maximum number of position in sessions.
            interest_extractor_num_heads (int): Number of heads in extractor layer.
            interest_interacting_hidden_size (int): Size of hidden units in interaction layer.
            interest_extractor_dropout (float, optional): Probability of Dropout in extractor layer. 
                Defaults to 0.0.
            interest_extractor_bias (bool, optional): Boolean flag to use bias variables in extractor layer. 
                Defaults to True.
            interest_interacting_num_layers (int): Number of hidden layers in interaction layer.
                Defaults to 1.
            interest_interacting_dropout (float, optional): Probability of Dropout in interaction layer.
                Defaults to 0.0.
            interest_interacting_bias (bool, optional): Boolean flag to use bias variables in interaction layer. 
                Defaults to True.
            use_bias_encoding (bool, optional): Boolean flag to use bias encoding. 
                Defaults to True.
        
        Attributes:
            bias_encoding (BiasEncodingLayer): Module of bias encoding layer.
            interest_extractor_layer (nn.Module): Module of interest extractor layer, Multihead Attention.
            interest_interacting_layer (nn.Module): Module of LSTM
        """
        # Refer to parent class
        super(DeepSessionInterestNetworkModel, self).__init__()

        # Initialize bias encoding variables
        if use_bias_encoding:
            self.bias_encoding = BiasEncodingLayer(
                embed_size       = embed_size,
                max_num_session  = max_num_session,
                max_num_position = max_num_position
            )

        else:
            self.register_parameter("bias_encoding", None)
        
        # Initialize multihead attention to be interest_extractor_layer
        self.interest_extractor_layer = nn.MultiheadAttention(
            embed_dim = embed_size,
            num_heads = interest_extractor_num_heads,
            dropout   = interest_extractor_dropout,
            bias      = interest_extractor_bias
        )

        # Initialize Bi LSTM to be interest_interacting_layer
        self.interest_interacting_layer = nn.LSTM(
            input_size    = embed_size,
            hidden_size   = interest_interacting_hidden_size,
            num_layers    = interest_interacting_num_layers,
            dropout       = interest_interacting_dropout,
            bias          = interest_interacting_bias,
            batch_first   = True,
            bidirectional = True
        )

        # Initialize adaptive average pooling layers
        self.interest_extractor_pooling = nn.AdaptiveAvgPool1d(1)
        self.interest_interacting_pooling = nn.AdaptiveAvgPool1d(1)
    
    def forward(self, session_embed_inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:

        # Add bias to session_embed_inputs from bias_encoding if use_bias_encoding
        # inputs: session_embed_inputs, shape = ((B, L, E), (B, ))
        # output: embed_inputs, shape = (B, L, E)
        if self.bias_encoding is not None:
            embed_inputs = self.bias_encoding(session_embed_inputs)
        else:
            embed_inputs = session_embed_inputs[0]
        
        # Apply self-attention to embed_inputs to extract session interest
        # inputs: embed_inputs, shape = (B, L, E)
        # output: extraction, shape = (B, L, E)
        embed_inputs = embed_inputs.rename(None)
        extraction, _ = self.interest_extractor_layer(embed_inputs, embed_inputs, embed_inputs)

        # Apply Bi-LSTM to attn_embed to calculate interaction between session interest
        interaction, _ = self.interest_interacting_layer(extraction)

        # Reshape and pooling outputs' from session extraction and interaction
        # inputs: extraction, shape = (B, L, E)
        # inputs: interaction, shape = (B, L, E)
        # output: pooled_extraction, shape = (B, E)
        # output: pooled_interaction, shape = (B, E)
        extraction.names = ("B", "L", "E")
        interaction.names = ("B", "L", "E")

        extraction = extraction.align_to("B", "E", "L")
        interaction = interaction.align_to("B", "E", "L")
        
        pooled_extraction = self.interest_extractor_pooling(extraction.rename(None))
        pooled_interaction = self.interest_interacting_pooling(interaction.rename(None))

        pooled_extraction.names = ("B", "E", "L")
        pooled_interaction.names = ("B", "E", "L")

        pooled_extraction = pooled_extraction.flatten(["E", "L"], "E")
        pooled_interaction = pooled_interaction.flatten(["E", "L"], "E")

        # Concatenate pooled_extraction and pooled_interaction before dense layer
        # inputs: pooled_extraction, shape = (B, E)
        # inputs: pooled_interaction, shape = (B, E)
        # output: features, shape = (B, E + E)
        # TODO: add dense features sparse features
        features = torch.cat([pooled_extraction, pooled_interaction], dim="E")

        # TODO: add dense layers and output (i.e. softmax) layer to complete the model.

        return features
