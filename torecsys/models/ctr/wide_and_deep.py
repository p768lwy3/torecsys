from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import MultilayerPerceptionLayer, WideLayer
from torecsys.models.ctr import CtrBaseModel


class WideAndDeepModel(CtrBaseModel):
    """
    Model class of Wide and Deep Model

    Wide and Deep Model is one of the most famous click through rate prediction model which is designed by Google in
    2016.

    :Reference:

    #. `Heng-Tze Cheng, 2016. Wide & Deep Learning for Recommender Systems <https://arxiv.org/pdf/1606.07792.pdf>`_.

    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 deep_layer_sizes: List[int],
                 out_dropout_p: Optional[float] = None,
                 wide_dropout_p: Optional[float] = None,
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize WideAndDeepModel

        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            deep_layer_sizes (List[int]): layer sizes of dense network
            out_dropout_p (float, optional): probability of Dropout in output layer. Defaults to None
            wide_dropout_p (float, optional): probability of Dropout in wide layer. Defaults to None
            deep_dropout_p (List[float], optional): probabilities of Dropout in dense network. Defaults to None
            deep_activation (torch.nn.Module, optional): activation function in dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        self.wide = WideLayer(
            inputs_size=num_fields,
            output_size=1,
            dropout_p=wide_dropout_p
        )
        self.deep = MultilayerPerceptionLayer(
            inputs_size=embed_size,
            output_size=1,
            layer_sizes=deep_layer_sizes,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )
        self.output = WideLayer(
            inputs_size=num_fields + 1,
            output_size=1,
            dropout_p=out_dropout_p
        )

    def forward(self, feat_inputs: torch.Tensor, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of WideAndDeepModel

        Args:
            feat_inputs (T), shape = (B, N, 1), data_type = torch.float: linear Features tensors
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors

        Returns:
            torch.Tensor, shape = (B, O), data_type = torch.float: output of WideAndDeepModel
        """
        # Name the feat_inputs tensor for flatten
        feat_inputs.names = ('B', 'N', 'E',)

        # Flatten feat_inputs to (B, N)
        feat_inputs = feat_inputs.flatten(('N', 'E',), 'N')

        # Calculate wide part
        # inputs: feat_inputs, shape = (B, N, E = 1)
        # output: wide_outputs, shape = (B, O = 1)
        wide_outputs = self.wide(feat_inputs)

        # Calculate deep part
        # inputs: emb_inputs, shape = (B, N, E)
        # output: deep_outputs, shape = (B, N, O = 1)
        deep_outputs = self.deep(emb_inputs)

        # Flatten deep_outputs to (B, N) and rename to (B, O)
        deep_outputs = deep_outputs.squeeze(-1)
        deep_outputs.names = ('B', 'O',)

        # Concatenate outputs of wide part and deep part
        # inputs: wide_outputs, shape = (B, 1)
        # inputs: deep_outputs, shape = (B, N)
        # output: outputs, shape = (B, N + 1)
        outputs = torch.cat([wide_outputs, deep_outputs], dim='O')

        # Linear before output
        # inputs: outputs, shape = (B, N + 1)
        # outputs: outputs, shape = (B, 1)
        outputs = self.output(outputs)

        # Since autograd does not support Named Tensor at this stage, drop the name of output tensor
        outputs = outputs.rename(None)

        return outputs
