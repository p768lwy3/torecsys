from . import _Inputs
import torch
import torch.nn as nn
from typing import List


class ImageInputs(_Inputs):
    r"""ImageInputs is a image input field to pass a image tensor, 
        and process with convalution neural network, and finally return a feed-forwarded features vectors
    """
    def __init__(self,
                 embed_size    : int,
                 in_channels   : int,
                 layers_size   : List[int],
                 kernels_size  : List[int],
                 strides       : List[int],
                 paddings      : List[int],
                 use_batchnorm : bool  = True,
                 dropout_p     : float = 0.0,
                 activation    : torch.nn.modules.activation = nn.ReLU()):
        r"""initialize the image inputs field
        
        Args:
            embed_size (int): embedding size
            in_channel (int): channel size of input
            layers_size (List[int]): layers size of convalution neural network
            kernels_size (List[int]): kernel size of convalution neural network
            strides (List[int]): strides of convalution neural network
            paddings (List[int]): paddings of convalution neural network
            use_batchnorm (bool, optional): boolean flag to use batch norm 2d after conv2d. Defaults to True.
            dropout_p (float, optional): dropout probability of dropout2d after conv2d or batchnorm2d. Defaults to 0.0.
            activation (torch.nn.modules.activation, optional): activation function after conv2d. Defaults to nn.ReLU().
        
        """
        super(ImageInputs, self).__init__()

        self.length = embed_size

        # set varialbes for the initialization of the model
        layers_size = [in_channels] + layers_size

        # initialize a modulist of convalution neural network
        self.model = nn.Sequential()
        iterations = enumerate(zip(layers_size[:-1], layers_size[1:], strides, paddings))
        for i, (in_c, out_c, k, s, p) in iterations:
            self.model.add_module("conv2d_%s" % i, nn.Convd2(in_c, out_c, kernel_size=k, stride=s, padding=p))
            if use_batchnorm:
                self.model.add_module("batchnorm2d_%s" % i, nn.BatchNorm2d(out_c))
            self.model.add_module("dropout2d_%s" % i, nn.Dropout2d(p=dropout_p))
            self.model.add_module("activation_%s" % i, activation)
        
        # using adaptive avg pooling and linear as a output layer
        self.model.add_module(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.model.add_module(nn.Linear(layers_size[-1], embed_size))
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Return features vectors of image inputs by convalution neural network
        
        Args:
            inputs (torch.Tensor), shape = (bacth size, number of channels, image height, image width), dtype = torch.float: image tensor
        
        Returns:
            torch.Tensor, shape = (batch size, 1, embedding size): features vectors
        """
        outputs = self.model(inputs)
        return outputs
