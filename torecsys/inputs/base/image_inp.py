from . import _Inputs
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
from typing import List


class ImageInputs(_Inputs):
    r"""Base Inputs class for image, which embed image by a stack of convalution neural network (CNN) 
    and fully-connect layer.
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
        r"""Initialize ImageInputs
        
        Args:
            embed_size (int): Size of embedding tensor
            in_channel (int): Number of channel of inputs
            layers_size (List[int]): Layers size of CNN
            kernels_size (List[int]): Kernels size of CNN
            strides (List[int]): Strides of CNN
            paddings (List[int]): Paddings of CNN
            pooling (str, optional): Method of pooling layer. 
                Defaults to avg_pooling.
            use_batchnorm (bool, optional): Whether batch normalization is applied after Conv2d. 
                Defaults to True.
            dropout_p (float, optional): Probability of Dropout2d. 
                Defaults to 0.0.
            activation (torch.nn.modules.activation, optional): Activation function of Conv2d. 
                Defaults to nn.ReLU().
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
