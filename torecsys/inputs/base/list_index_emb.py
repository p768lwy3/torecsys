from . import _Inputs
from torecsys.functional import show_attention, dummy_attention
from torecsys.utils.decorator import jit_experimental, no_jit_experimental
from functools import partial
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


class ListIndexEmbedding(_Inputs):
    r"""ListIndexEmbedding is a embedding field to pass a list of index without order 
        process by Multihead Attention, and finally return aggregated embedding tensor
    """
    @jit_experimental
    def __init__(self,
                 embed_size    : int,
                 field_size    : int,
                 padding_idx   : int  = 0,
                 use_attn      : bool = True,
                 output_method : str  = "avg_pooling",
                 nn_embedding  : nn.Parameter = None,
                 **kwargs):
        r"""initialize the list index embedding field
        
        Args:
            embed_size (int): embedding size
            field_size (int): field size
            padding_idx (int, optional): padding index of field. Defaults to 0.
            use_attn (bool, optional): boolean flag to use multihead attention Defaults to True.
            output_method (str, optional): string of aggregation method, Allow: ["avg_pooling", "max_pooling", "mean", "sum"]. Defaults to "avg_pooling".
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None.
        
        Kwargs:
            num_heads (int): Number of heads for MultiheadAttention
            dropout (float, optional): Dropout probability for MultiheadAttention. Default = 0.0.
            bias (bool, optional): boolean flag of bias. Default = True.
            add_bias_kv (bool, optional): boolean flag of add_bias_kv. Defalut = False.
            add_zero_attn (bool, optional): boolean flag of add_zero_attn. Default = False.
        
        Raises:
            ValueError: when output_method is not in ["avg_pooling", "max_pooling", "mean", "sum"].
        """
        super(ListIndexEmbedding, self).__init__()

        self.embed_size = embed_size
        self.field_size = field_size
        self.length = embed_size

        if nn_embedding is not None:
            self.embed_size = nn_embedding.size(1)
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        else:
            self.embed_size = embed_size
            self.embedding = nn.Embedding(field_size, embed_size, padding_idx=padding_idx)
        
        # initialize Attention layer
        self.use_attn = use_attn
        if self.use_attn:
            # parse arguments of Attention layer
            self.attn_args = dict(
                embed_dim     = embed_size,
                num_heads     = kwargs.get("num_heads"),
                dropout       = kwargs.get("dropout", 0.0),
                bias          = kwargs.get("bias", True),
                add_bias_kv   = kwargs.get("add_bias_kv", False),
                add_zero_attn = kwargs.get("add_zero_attn", False)
            )
            self.attention = nn.MultiheadAttention(**self.attn_args)
        else:
            # set self.attention to dummy_attention which will return key directly
            self.attention = dummy_attention
        
        # initialize the output layer
        __output_method__ = ["avg_pooling", "max_pooling", "mean", "none", "sum"]
        if output_method == "avg_pooling":
            self.output_layer = nn.AdaptiveAvgPool1d(1)
        elif output_method == "max_pooling":
            self.output_layer = nn.AdaptiveMaxPool1d(1)
        elif output_method == "mean":
            self.output_layer = partial(torch.mean, dim=1, keepdim=True)
        elif output_method == "none":
            self.output_layer = torch.Tensor
        elif output_method == "sum":
            self.output_layer = partial(torch.sum, dim=1, keepdim=True)
        else:
            raise ValueError("output_method only allows [%s]." % (", ".join(__output_method__)))
        self.output_method = output_method

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Return aggregated embedding vectors of inputs by self-attention model
        
        Notations:
            B: batch size
            L: max sequence length
            E: embedding size

        Args:
            inputs (T), shape = (B, L), dtype = torch.long: list of indices to be embedded
        
        Returns:
            Tuple[T, T], shape = ((B, 1 or L, E), (B, L, L) or (None)), dtype = (torch.float, torch.float): (aggregated) embedding vectors and attention weights
        """
        # get embedding and output shape = (B, L, E) then, transpose them to (L, B, E)
        outputs = self.embedding(inputs)
        outputs = outputs.transpose(0, 1)

        # compute self-attention,
        # outputs' shape = (B, L, E) and attn_weights' shape = (B, L, L) if use_attn = True
        # then transpose back to (B, L, E)
        outputs, attn_weights = self.attention(outputs, outputs, outputs)
        outputs = outputs.transpose(0, 1)
        
        # calculate aggregation of outputs
        if self.output_method == "avg_pooling" or self.output_method == "max_pooling":
            # transpose from (B, L, E) to (B, E, L)
            outputs = outputs.transpose(1, 2)
            # shape of outputs = (B, E, 1)
            outputs = self.output_layer(outputs)
            # transpose from (B, E, 1) to (B, 1, E)
            outputs = outputs.transpose(1, 2)
        else:
            # outputs' shape = (B, 1, E) if output_method == "mean" or "sum"
            # else outputs' shape = (B, L, E) if output_method == "none"
            outputs = self.output_layer(outputs)
        return outputs, attn_weights

    @no_jit_experimental
    def show_attention(self, 
                       inputs  : torch.Tensor, 
                       savedir : str = None):
        
        if self.use_attn:
            if inputs.size(0) != 1:
                raise ValueError("Only allow to pass one sample for visualization of attention at this stage")
            
            with torch.no_grad():
                # get embedding vectors first
                # and transpose from (1, L, E) to (L, 1, E)
                outputs = self.embedding(inputs)
                outputs = outputs.transpose(0, 1)
                
                # calcualte self-attentions, and attentions' shape = (1, L, L)
                _, attn_weights = self.attention(outputs, outputs, outputs)
            
            # flatten the attentions by removing 1st dimension
            attention = np.squeeze(attn_weights.numpy(), axis=0)

            # create a list of string of index from inputs to be the axis of plot
            axis = [str(x) for x in inputs.squeeze().tolist()]

            # show attentions with a heatmap plot
            show_attention(attention, xaxis=axis, yaxis=axis, savedir=savedir)
        
        else:
            raise ValueError("show_attention cannot be called if use_attn is False.")
