import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from torecsys.layers import BaseLayer
from torecsys.utils.operations import combination


class FieldAllTypeBilinear(BaseLayer):
    r"""
    Applies a bilinear transformation to the incoming data: :math:`y = x_1 \cdot W \odot x_2 + b`
    
    Args:
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        bias: If set to False, the layer will not learn an additive bias. Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}` and
            :math:`*` means any number of additional dimensions. All but the last dimension
            of the inputs should be the same
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`.
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
            and all but the last dimension are the same shape as the input
    
    Examples::

        >>> m = FieldAllTypeBilinear(20, 20)
        >>> input1 = torch.randn(128, 10, 20)
        >>> input2 = torch.randn(128, 10, 3)
        >>> output = m(input1, input2)
        >>> print(output.size())
            torch.Size([128, 10, 3])
    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs1': ('B', 'NC2', 'E',),
            'inputs2': ('B', 'NC2', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', 'NC2', 'E',)
        }

    __constants__ = ['in1_features', 'in2_features', 'bias']

    def __init__(self, in1_features, in2_features, bias=True):
        super(FieldAllTypeBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.weight = nn.Parameter(torch.Tensor(in1_features, in2_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(in2_features))
        else:
            self.register_parameter('bias', nn.Parameter(torch.tensor([0])))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.shape[0])
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        output = torch.mul(torch.matmul(input1, self.weight), input2)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        return f'in1_features={self.in1_features}, in2_features={self.in2_features}, bias={self.bias is not None}'


class FieldEachTypeBilinear(BaseLayer):
    r"""
    Applies a bilinear transformation to the incoming data: :math:`y = x_1 \cdot W \odot x_2 + b`
    
    Args:
        in_features: size of first dimension in first input sample and second input sample
        in1_features: size of each first input sample
        in2_features: size of each second input sample
        bias: If set to False, the layer will not learn an additive bias. Default: ``True``

    Shape:
        - Input1: :math:`(N, *, H_{in1})` where :math:`H_{in1}=\text{in1\_features}` and
              :math:`*` means any number of additional dimensions. All but the last dimension
              of the inputs should be the same
        - Input2: :math:`(N, *, H_{in2})` where :math:`H_{in2}=\text{in2\_features}`
        - Output: :math:`(N, *, H_{out})` where :math:`H_{out}=\text{out\_features}`
          and all but the last dimension are the same shape as the input
    
    Examples::

        >>> m = FieldAllTypeBilinear(20, 20)
        >>> input1 = torch.randn(128, 10, 20)
        >>> input2 = torch.randn(128, 10, 3)
        >>> output = m(input1, input2)
        >>> print(output.size())
            torch.Size([128, 10, 3])
    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs1': ('B', 'NC2', 'E',),
            'inputs2': ('B', 'NC2', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'outputs': ('B', 'NC2', 'E',)
        }

    __constants__ = ['in_features', 'in1_features', 'in2_features', 'bias']

    def __init__(self, in_features, in1_features, in2_features, bias=True):
        super(FieldEachTypeBilinear, self).__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.weight = nn.Parameter(torch.Tensor(in_features, in1_features, in2_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(in_features, in2_features))
        else:
            self.register_parameter('bias', nn.Parameter(torch.tensor([0])))

        self.reset_parameters()

    def reset_parameters(self):
        bound = 1 / math.sqrt(self.weight.shape[0])
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input1, input2):
        output = torch.matmul(input1.unsqueeze(-2), self.weight).squeeze(-2)
        output = torch.mul(output, input2)
        if self.bias is not None:
            output += self.bias
        return output

    def extra_repr(self):
        return f'in1_features={self.in1_features}, in2_features={self.in2_features}, bias={self.bias is not None}'


class BilinearInteractionLayer(BaseLayer):
    """
    Layer class of Bilinear-Interaction.

    Bilinear-Interaction layer is used in FiBiNet proposed by `Tongwen Huang et al`[1] to combine inner-product and
    Hadamard product to learn features' interactions with extra parameters W.

    :Reference:

    #. `Tongwen Huang et al, 2019. FibiNET: Combining Feature Importance and Bilinear feature Interaction for
    Click-Through Rate Prediction <https://arxiv.org/abs/1905.09433>`_.
     
    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'N', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return self.bilinear.outputs_size

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 bilinear_type: str = 'all',
                 bias: bool = True):
        """
        Initialize BilinearInteractionLayer
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            bilinear_type (str, optional): type of bilinear to calculate interactions. Defaults to "all"
            bias (bool, optional): flag to control using bias. Defaults to True
        
        Raises:
            NotImplementedError: /
            ValueError: when bilinear_type is not in ["all", "each", "interaction"]
        """
        super().__init__()

        self.row_idx = []
        self.col_idx = []
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.row_idx.append(i)
                self.col_idx.append(j)
        self.row_idx = torch.LongTensor(self.row_idx)
        self.col_idx = torch.LongTensor(self.col_idx)

        num_interaction = combination(num_fields, 2)

        self.bilinear_type = bilinear_type
        if bilinear_type == 'all':
            self.bilinear = FieldAllTypeBilinear(embed_size, embed_size, bias=bias)
        elif bilinear_type == 'each':
            self.bilinear = FieldEachTypeBilinear(num_interaction, embed_size, embed_size, bias=bias)
        elif bilinear_type == 'interaction':
            # self.bilinear = FieldInteractionTypeBilinear(num_interaction, embed_size, embed_size, bias=bias)
            raise NotImplementedError()
        else:
            raise ValueError('bilinear_type only allows: ["all", "each", "interaction"].')

    def extra_repr(self) -> str:
        """
        Return information in print-statement of layer
        
        Returns:
            str: Information of print-statement of layer
        """
        return f'bilinear_type={self.bilinear_type}'

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of BilinearInteractionLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: Embedded features tensors
        
        Returns:
            T, shape = (B, NC2, E), data_type = torch.float: Output of BilinearInteractionLayer
        """
        # Indexing data for bilinear interaction
        # inputs: emb_inputs, shape = (B, N, E)
        # output: p, shape = (B, NC2, E)
        # output: q, shape = (B, NC2, E)
        p = emb_inputs.rename(None)[:, self.row_idx]
        q = emb_inputs.rename(None)[:, self.col_idx]

        # Calculate bilinear interaction with index slicing
        # inputs: p, shape = (B, NC2, E)
        # inputs: q, shape = (B, NC2, E)
        # output: output, shape = (B, NC2, E)
        output = self.bilinear(p, q)

        # Rename tensor names
        output.names = ('B', 'N', 'O',)
        return output
