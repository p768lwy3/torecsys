from typing import Optional

import torch
import torch.nn as nn
import torchvision

from torecsys.inputs.base import BaseInput


class PretrainedImageInput(BaseInput):
    """
    Base Input class for image, which embed by famous pretrained model in Computer Vision.
    """

    def __init__(self,
                 embed_size: int,
                 model_name: str,
                 pretrained: Optional[bool] = True,
                 no_grad: Optional[bool] = False,
                 verbose: Optional[bool] = False,
                 device: str = 'cpu'):
        """
        Initialize PretrainedImageInput
        
        Args:
            embed_size (int): size of embedding tensor
            model_name (str): model name. Refer to: `torchvision.models
                <https://pytorch.org/vision/stable/models.html>`_
            pretrained (bool, optional): whether use pre-trained model or not
                Defaults to True
            verbose (bool, optional): whether display progress bar of the download or not
                Defaults to False
            no_grad (bool, optional): whether parameters in pre-trained model
                (excluding fc, i.e. output nn.Linear layer) is set to no gradients or not
                Defaults to False
            device (str): device of torch. Default to cpu.

        :Reference:

        #. `Docs of torchvision.models <https://pytorch.org/docs/stable/torchvision/models.html>`_

        """
        super().__init__()

        self.length = embed_size
        self.model = getattr(torchvision.models, model_name)(pretrained=pretrained, progress=verbose)

        if model_name in ['vgg16', 'vgg19']:
            classifier = self.model.classifier
            last_in_size = classifier[-1].in_features
            classifier[-1] = nn.Linear(last_in_size, embed_size)
        else:
            last_in_size = self.model.fc.in_features
            self.model.fc = nn.Linear(last_in_size, embed_size)

        if no_grad:
            for name, param in self.model.named_parameters():
                if not name.startswith('fc'):
                    param.requires_grad = False

        self.model = self.model.to(device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of PretrainedImageInput
        
        Args:
            inputs (T), shape = (B, C, H_{i}, W_{i}), data_type = torch.float: tensor of images.
        
        Returns:
            T, shape = (B, 1, E): output of PretrainedImageInput.
        """
        outputs = self.model(inputs.rename(None))

        if outputs.dim() == 2:
            outputs = outputs.unsqueeze(dim=1)

        outputs.names = ('B', 'N', 'E',)
        return outputs
