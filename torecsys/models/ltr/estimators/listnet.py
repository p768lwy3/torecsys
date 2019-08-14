from . import _LtrEstimator
import torch
import torch.nn as nn
from typing import Callable, List


class ListNet(_LtrEstimator):
    r"""ListNet is a model which is a stack of feed forward dense layers to predict ranking of the
    given list, e.g. :math:`y_{true} = [1, 4, 5, 6, 3, 2]` .

    :Reference:
    
    #. `Zhe Cao et al, 2007. Learning to Rank: From Pairwise Approach to Listwise Approach <https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2007-40.pdf>`

    """
    def __init__(self,
                 num_fields : int,
                 layer_sizes: List[int],
                 output_size: int = 1,
                 dropout_p  : List[float] = None,
                 activation : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""initialize list net
        
        Args:
            num_fields (int): number of fields in inputs, i.e. inputs' dimension of 1st-Linear layer
            layer_sizes (List[int]): layer sizes of Linear layers
            output_size (int, optional): output size of list net. Defaults to 1.
            dropout_p (List[float]): dropout probability after activation of each layer. Allow: [None, list of float for each layer]. Defaults to None.
            activation (Callable[[torch.Tensor], torch.Tensor], optional): activation function of each layer. Allow: [None, Callable[[torch.Tensor], torch.Tensor]]. Defaults to nn.ReLU().
        
        Raises:
            ValueError: when dropout_p is not None and length of dropout_p is not equal to that of layer_sizes
        """
        # initialize nn.Module class
        super(ListNet, self).__init__()

        # check if length of dropout_p is not equal to length of layer_sizes
        if dropout_p is not None and len(dropout_p) != len(layer_sizes):
            raise ValueError("length of dropout_p must be equal to length of layer_sizes.")
        
        self.model = nn.Sequential()
        layer_sizes = [num_fields] + layer_sizes + [output_size]
        for i, (in_d, out_d) in enumerate(zip(layer_sizes[:-1], layer_sizes[1])):
            self.model.add_module("linear_%s" % i, nn.Linear(in_d, out_d))
            # add activation layer after linear, excluding the last layer
            if activation is not None and i != len(layer_sizes) - 1:
                self.model.add_module("activation_%s" % i, activation)
            # add dropout layer after linear, excluding the last layer
            if dropout_p is not None and i != len(layer_sizes) - 1:
                self.model.add_module("dropout_%s" % i, nn.Dropout(dropout_p[i]))
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""feed-forward calculation of ListNet
        
        Args:
            inputs (torch.Tensor), shape = (batch size, max sequence length, number of fields), dtype = torch.float: inputs' features of ranking list
        
        Returns:
            torch.Tensor, shape = (batch size, max sequence length, output size), dtype = torch.float: ranking of list calculated by the model
        """
        outputs = self.model(inputs)
        return outputs
