from . import _Inputs
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import List


class ImageInputs(_Inputs):
    r"""ImageInputs is a image input field to pass a image tensor, 
    and process with convalution neural network, and finally return a feed-forwarded features vectors
    """
    @jit_experimental
    def __init__(self,
                 embed_size    : int,
                 in_channels   : int,
                 layers_size   : List[int],
                 kernels_size  : List[int],
                 strides       : List[int],
                 paddings      : List[int],
                 pooling       : str = "avg_pooling",
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
            pooling (str, optional): pooling layer method. Defaults to avg_pooling.
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
        iterations = enumerate(zip(layers_size[:-1], layers_size[1:], kernels_size, strides, paddings))
        for i, (in_c, out_c, k, s, p) in iterations:
            self.model.add_module("conv2d_%s" % i, nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p))
            if use_batchnorm:
                self.model.add_module("batchnorm2d_%s" % i, nn.BatchNorm2d(out_c))
            self.model.add_module("dropout2d_%s" % i, nn.Dropout2d(p=dropout_p))
            self.model.add_module("activation_%s" % i, activation)
        
        # using adaptive avg pooling and linear as a output layer
        if pooling == "max_pooling":
            self.model.add_module("pooling", nn.AdaptiveMaxPool2d(output_size=(1, 1)))
        elif pooling == "avg_pooling":
            self.model.add_module("pooling", nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        
        self.fc = nn.Linear(layers_size[-1], embed_size)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Return features vectors of image inputs by convalution neural network
        
        Args:
            inputs (T), shape = (B, C, H_{i}, W_{i}), dtype = torch.float: image tensor
        
        Returns:
            T, shape = (B, 1, E): features vectors
        """
        # output's shape of convolution model = (B, H_{last}, 1, 1)
        outputs = self.model(inputs)
        
        # output's shape of fully-connect layers = (B, E)
        outputs = self.fc(outputs.squeeze())

        return outputs.unsqueeze(1)
