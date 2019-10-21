from . import _Inputs
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn
from typing import List


class ImageInputs(_Inputs):
    r"""Base Inputs class for image, which embed image by a stack of convalution neural network (CNN) 
    and fully-connect layer.
    """
    @no_jit_experimental_by_namedtensor
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
        r"""Initialize ImageInputs.
        
        Args:
            embed_size (int): Size of embedding tensor
            in_channel (int): Number of channel of inputs
            layers_size (List[int]): Layers size of CNN
            kernels_size (List[int]): Kernels size of CNN
            strides (List[int]): Strides of CNN
            paddings (List[int]): Paddings of CNN
            pooling (str, optional): Method of pooling layer. 
                Defaults to avg_pooling.
            use_batchnorm (bool, optional): Whether batch normalization is applied or not after Conv2d. 
                Defaults to True.
            dropout_p (float, optional): Probability of Dropout2d. 
                Defaults to 0.0.
            activation (torch.nn.modules.activation, optional): Activation function of Conv2d. 
                Defaults to nn.ReLU().
        
        Attributes:
            length (int): Size of embedding tensor.
            model (torch.nn.Sequential): Sequential of CNN-layers.
            fc (torch.nn.Module): Fully-connect layer (i.e. Linear layer) of outputs.
        
        Raises:
            ValueError: when pooling is not in ["max_pooling", "avg_pooling"]
        """
        # refer to parent class
        super(ImageInputs, self).__init__()

        # bind length to embed_size
        self.length = embed_size

        # stack in_channels and layers_size
        layers_size = [in_channels] + layers_size

        # initialize sequential for CNN-layers
        self.model = nn.Sequential()

        # create a generator top create CNN-layers
        iterations = enumerate(zip(layers_size[:-1], layers_size[1:], kernels_size, strides, paddings))
        
        # loop through iterations to create CNN-layers
        for i, (in_c, out_c, k, s, p) in iterations:
            # add Conv2d to sequential
            self.model.add_module("conv2d_%s" % i, nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p))
            
            # add batch norm to sequential
            if use_batchnorm:
                self.model.add_module("batchnorm2d_%s" % i, nn.BatchNorm2d(out_c))
            
            # add dropout to sequential
            self.model.add_module("dropout2d_%s" % i, nn.Dropout2d(p=dropout_p))

            # add activation to sequential
            self.model.add_module("activation_%s" % i, activation)
        
        # add pooling to sequential
        if pooling == "max_pooling":
            self.model.add_module("pooling", nn.AdaptiveMaxPool2d(output_size=(1, 1)))
        elif pooling == "avg_pooling":
            self.model.add_module("pooling", nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        else:
            raise ValueError('pooling must be in ["max_pooling", "avg_pooling"].')
        
        # initialize linear layer of fully-connect outputs
        self.fc = nn.Linear(layers_size[-1], embed_size)
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of ImageInputs.
        
        Args:
            inputs (T), shape = (B, C, H_{i}, W_{i}), dtype = torch.float: Tensor of images.
        
        Returns:
            T, shape = (B, 1, E): Output of ImageInputs.
        """
        # feed forward to sequential of CNN,
        # output's shape of convolution model = (B, C_{last}, 1, 1)
        outputs = self.model(inputs.rename(None))
        outputs.names = ("B", "C", "H", "W")
        
        # fully-connect layer to transform outputs,
        # output's shape of fully-connect layers = (B, E)
        outputs = self.fc(outputs.rename(None).squeeze())
        
        # unsqueeze the outputs in dim = 1 and set names to the tensor,
        outputs = outputs.unsqueeze(1)
        outputs.names = ("B", "N", "E")

        return outputs
