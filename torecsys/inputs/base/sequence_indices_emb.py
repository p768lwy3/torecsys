from collections import namedtuple
from functools import partial
from typing import Optional

import torch
import torch.nn as nn

from torecsys.inputs.base import BaseInput


class SequenceIndicesEmbedding(BaseInput):
    """
    Base Input class for embedding of sequence of indices with order, which embed the sequence by
    Recurrent Neural Network (RNN) and aggregate before return.
    """

    def __init__(self,
                 embed_size: int,
                 field_size: int,
                 padding_idx: Optional[int] = 0,
                 rnn_method: Optional[str] = 'lstm',
                 output_method: Optional[str] = 'avg_pooling',
                 nn_embedding: Optional[nn.Parameter] = None,
                 **kwargs):
        """
        Initialize SequenceIndicesEmbedding.
        
        Args:
            embed_size (int): size of embedding tensor
            field_size (int): size of inputs field
            padding_idx (int, optional): padding index. Defaults to 0
            rnn_method (str, optional): method of RNN. allow: ["gru", "lstm", "rnn"].
                Defaults to "lstm".
            output_method (str, optional): method of aggregation.
                allow: ["avg_pooling", "max_pooling", "mean", "none", "sum"].
                Defaults to "avg_pooling".
            nn_embedding (nn.Parameter, optional): pretrained embedding values. Defaults to None.
            
        Kwargs:
            num_layers (int): number of layers of RNN. Default to 1.
            bias (bool): whether bias is added to RNN or not. Default to True.
            dropout (float): probability of Dropout in RNN. Default to 0.0.
            bidirectional (bool): whether bidirectional is used in RNN or not. Default to False.
        
        Attributes:
            length (int): size of embedding tensor.
            embedding (torch.nn.Module): embedding layer.
            rnn_layers (torch.nn.Module): rnn layers.
            aggregation (Union[torch.nn.Module, callable]): pooling layer or aggregation function.
            output_method (string): type of output_method.
        
        Raises:
            ValueError: when rnn_method is not in ["gru", "lstm", "rnn"].
            ValueError: when output_method is not in ["avg_pooling", "max_pooling", "mean", "sum"].
        """
        super().__init__()

        if nn_embedding is not None:
            self.length = nn_embedding.size('E')
            self.embedding = nn.Embedding.from_pretrained(nn_embedding)
        else:
            self.length = embed_size
            self.embedding = nn.Embedding(field_size, embed_size, padding_idx=padding_idx, **kwargs)

        bidirectional = kwargs.get('bidirectional', False)
        hidden_size = embed_size // 2 if bidirectional else embed_size
        rnn_args = {
            'input_size': embed_size,
            'hidden_size': hidden_size,
            'num_layers': kwargs.get('num_layers', 1),
            'bias': kwargs.get('bias', True),
            'batch_first': True,
            'dropout': kwargs.get('dropout', 0.0),
            'bidirectional': bidirectional
        }

        if rnn_method == 'rnn':
            self.rnn_layers = nn.RNN(**rnn_args)
        elif rnn_method == 'lstm':
            self.rnn_layers = nn.LSTM(**rnn_args)
        elif rnn_method == 'gru':
            self.rnn_layers = nn.GRU(**rnn_args)
        else:
            raise ValueError('rnn_method only allows ["rnn", "lstm", "gru"].')

        self.output_method = output_method
        if output_method == 'avg_pooling':
            self.aggregation = nn.AdaptiveAvgPool1d(1)
        elif output_method == 'max_pooling':
            self.aggregation = nn.AdaptiveMaxPool1d(1)
        elif output_method == 'mean':
            self.aggregation = partial(torch.mean, dim='N', keepdim=True)
        elif output_method == 'none':
            self.aggregation = torch.Tensor
        elif output_method == 'sum':
            self.aggregation = partial(torch.sum, dim='N', keepdim=True)
        else:
            raise ValueError('output_method only allows ["avg_pooling", "max_pooling", "mean", "none", "sum"].')

    def set_schema(self, inputs: str, **kwargs):
        """
        Initialize input layer's schema of SequenceIndicesEmbedding
        
        Args:
            inputs (str): String of input's field name
            kwargs.lengths (str): String of length's field name
        """
        lengths = kwargs.get('lengths', None)
        if lengths is None:
            raise ValueError('')

        schema = namedtuple('Schema', ['inputs', 'lengths'])
        self.schema = schema(inputs=[inputs], lengths=lengths)

    def forward(self, inputs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of SequenceIndicesEmbedding

        Args:
            inputs (T), shape = (B, L), data_type = torch.long: sequence of tensor of indices in inputs fields
            lengths (T), shape = (B), data_type = torch.long: length of sequence of tensor of indices
        
        Returns:
            T, shape = (B, 1 or L, E): outputs of SequenceIndicesEmbedding
        """
        # sort inputs with the lengths
        lengths, perm_idx = lengths.rename(None).sort(0, descending=True)
        inputs = inputs.rename(None)[perm_idx]

        # sort for the desort index
        _, desort_idx = perm_idx.sort()

        # get embedded vectors
        embedded = self.embedding(inputs)

        # pack_padded, where packed_outputs' shape = (B + L, E)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu().numpy(), batch_first=True)

        # forward calculate of rnn
        rnn_outputs, state = self.rnn_layers(packed)

        # unpack output with shape = (B, L, E)
        unpacked, _ = torch.nn.utils.rnn.pad_packed_sequence(rnn_outputs, batch_first=True)

        # desort the order of samples
        outputs = unpacked[desort_idx]

        # calculate aggregation of outputs
        if self.output_method in ['avg_pooling' or 'max_pooling']:
            # transpose from (B, L, E) to (B, E, L)
            # outputs = outputs.transpose(1, 2)
            outputs.names = ('B', 'L', 'E',)
            outputs = outputs.align_to('B', 'E', 'L')

            # shape of outputs = (B, E, 1)
            outputs = self.aggregation(outputs.rename(None))

            # transpose from (B, E, 1) to (B, 1, E)
            # outputs = outputs.transpose(1, 2)
            outputs.names = ('B', 'E', 'N',)
            outputs = outputs.align_to('B', 'N', 'E')

        else:
            # outputs' shape = (B, 1, E) if output_method == "mean" or "sum"
            # else outputs' shape = (B, L, E) if output_method == "none"
            outputs = self.aggregation(outputs.rename(None))
            outputs.names = ('B', 'N', 'E')

        return outputs
