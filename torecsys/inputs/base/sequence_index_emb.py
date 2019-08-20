from . import _Inputs
from torecsys.utils.decorator import jit_experimental
from functools import partial
import torch
import torch.nn as nn


class SequenceIndexEmbedding(_Inputs):
    r"""SequenceIndexEmbedding is a embedding field to pass a sequence of index with order
    process by Recurrent Neural Network, and finally return aggregated embedding tensor"""
    @jit_experimental
    def __init__(self,
                 embed_size   : int,
                 field_size   : int,
                 padding_idx  : int = 0,
                 rnn_method   : str = "lstm",
                 output_method: str = "avg_pooling",
                 nn_embedding : nn.Parameter = None,
                 **kwargs):
        r"""initialize the sequence index embedding field
        
        Args:
            embed_size (int): embedding size
            field_size (int): field size
            padding_idx (int, optional): padding index of field. Defaults to 0.
            rnn_method (str, optional): string of rnn method, Allow: ["gru", "lstm", "rnn"]. Defaults to "lstm".
            output_method (str, optional): string of aggregation method, Allow: ["avg_pooling", "max_pooling", "mean", "sum"]. Defaults to "avg_pooling".
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None.
        
        Kwargs:
            num_layers (int): number of layers of recurrent neural network
            bias (bool): boolean flag to use bias variable in recurrent neural network
            dropout (float): dropout ratio of recurrent neural network
            bidirectional (bool): boolean flag to use bidirectional recurrent neural network
        
        Raises:
            ValueError: when rnn_method is not in ["gru", "lstm", "rnn"].
            ValueError: when output_method is not in ["avg_pooling", "max_pooling", "mean", "sum"].
        """
        super(SequenceIndexEmbedding, self).__init__()
        
        if nn_embedding is not None:
            self.length = nn_embedding.size(1)
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        else:
            self.length = embed_size
            self.embedding = nn.Embedding(field_size, embed_size, padding_idx=padding_idx, **kwargs)

        __rnn_method__ = ["rnn", "lstm", "gru"]
        bidirectional = kwargs.get("bidirectional", False)
        if bidirectional:
            hidden_size = embed_size // 2
        else:
            hidden_size = embed_size
        
        rnn_args = dict(
            input_size    = embed_size,
            hidden_size   = hidden_size,
            num_layers    = kwargs.get("num_layers", 1),
            bias          = kwargs.get("bias", True),
            batch_first   = True,
            dropout       = kwargs.get("dropout", 0.0),
            bidirectional = bidirectional
        )

        if rnn_method == "rnn":
            self.rnn_layers = nn.RNN(**rnn_args)
        elif rnn_method == "lstm":
            self.rnn_layers = nn.LSTM(**rnn_args)
        elif rnn_method == "gru":
            self.rnn_layers = nn.GRU(**rnn_args)
        else:
            raise ValueError("rnn_method only allows [%s]." % (", ".join(__rnn_method__)))

        __output_method__ = ["avg_pooling", "max_pooling", "mean", "none", "sum"]
        
        self.output_method = output_method
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
    
    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        r"""Return aggregated embedding vectors of inputs which passed over Recurrent Neural Network, and take aggregation after that.

        Args:
            inputs (T), shape = (B, L), dtype = torch.long: sequence of indices to be embedded
            lengths (T), shape = (B, ), dtype = torch.long: length of inputs sequence
        
        Returns:
            T, shape = (B, 1 or L, E): (aggregated) embedding vectors
        """
        # sort inputs with the lengths
        lengths, perm_idx = lengths.sort(0, descending=True)
        inputs = inputs[perm_idx]
        
        # sort for the desort index
        _, desort_idx = perm_idx.sort()

        # get embedded vectors
        embedded = self.embedding(inputs)

        # pack_padded, where packed_outputs' shape = (B + L, E)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu().numpy(), batch_first=True)
        
        # forward calculate of LSTM
        rnn_outputs, state = self.rnn_layers(packed)
        
        # unpack output with shape = (B, L, E)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)
        
        # desort the order of samples
        outputs = unpacked[desort_idx]

        # calculate aggregation of outputs
        if self.output_method in ["avg_pooling" or "max_pooling"]:
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
        return outputs
