from typing import List, Optional, TypeVar

import torch
import torch.nn as nn

from torecsys.inputs.base import BaseInput


class ImageInput(BaseInput):
    """
    Base Input class for image, which embed image by a stack of convolution neural network (CNN)
    and fully-connect layer.
    """
    ImageInputs = TypeVar('ImageInput')

    def __init__(self,
                 embed_size: int,
                 in_channels: int,
                 layers_size: List[int],
                 kernels_size: List[int],
                 strides: List[int],
                 paddings: List[int],
                 pooling: Optional[str] = 'avg_pooling',
                 use_batchnorm: Optional[bool] = True,
                 dropout_p: Optional[float] = 0.0,
                 activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize ImageInput.
        
        Args:
            embed_size (int): Size of embedding tensor
            in_channels (int): Number of channel of inputs
            layers_size (List[int]): Layers size of CNN
            kernels_size (List[int]): Kernels size of CNN
            strides (List[int]): Strides of CNN
            paddings (List[int]): Paddings of CNN
            pooling (str, optional): Method of pooling layer
                Defaults to avg_pooling
            use_batchnorm (bool, optional): Whether batch normalization is applied or not after Conv2d
                Defaults to True
            dropout_p (float, optional): Probability of Dropout2d
                Defaults to 0.0
            activation (torch.nn.modules.activation, optional): Activation function of Conv2d
                Defaults to nn.ReLU()
        
        Raises:
            ValueError: when pooling is not in ["max_pooling", "avg_pooling"]
        """
        super().__init__()

        self.length = embed_size
        self.model = nn.Sequential()

        layers_size = [in_channels] + layers_size
        iterations = enumerate(zip(layers_size[:-1], layers_size[1:], kernels_size, strides, paddings))

        for i, (in_c, out_c, k, s, p) in iterations:
            conv2d_i = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p)
            self.model.add_module(f'conv2d_{i}', conv2d_i)

            if use_batchnorm:
                self.model.add_module(f'batchnorm2d_{i}', nn.BatchNorm2d(out_c))

            self.model.add_module(f'dropout2d_{i}', nn.Dropout2d(p=dropout_p))
            self.model.add_module(f'activation_{i}', activation)

        if pooling == 'max_pooling':
            pooling_layer = nn.AdaptiveMaxPool2d(output_size=(1, 1,))
        elif pooling == 'avg_pooling':
            pooling_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1,))
        else:
            raise ValueError('pooling must be in ["max_pooling", "avg_pooling"].')
        self.model.add_module('pooling', pooling_layer)

        self.fc = nn.Linear(layers_size[-1], embed_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of ImageInput
        
        Args:
            inputs (torch.tensor), shape = (B, C, H_{i}, W_{i}), data_type = torch.float: tensor of images
        
        Returns:
            torch.tensor, shape = (B, 1, E): output of ImageInput
        """
        # output's shape of convolution model = (B, C_{last}, 1, 1)
        outputs = self.model(inputs.rename(None))
        outputs.names = ('B', 'C', 'H', 'W',)

        # output's shape of fully-connect layers = (B, E)
        outputs = self.fc(outputs.rename(None).squeeze())

        # unsqueeze the outputs in dim = 1 and set names to the tensor,
        outputs = outputs.unsqueeze(1)
        outputs.names = ('B', 'N', 'E',)
        return outputs
