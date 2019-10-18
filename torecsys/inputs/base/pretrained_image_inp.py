from . import _Inputs
from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn
import torchvision


class PretrainedImageInputs(_Inputs):
    r"""Base Inputs class for image, which embed by famous pretrained model in Computer 
    Vision.
    """
    @jit_experimental
    def __init__(self,
                 embed_size : int,
                 model_name : str,
                 pretrained : bool = True,
                 progress   : bool = False,
                 no_grad    : bool = False):
        r"""Initialize PretrainedImageInputs.
        
        Args:
            embed_size (int): Size of embedding tensor
            model_name (str): Model name. Refer to: `torchvision.models <https://pytorch.org/docs/stable/torchvision/models.html>`_.
            pretrained (bool, optional): Whether use pretrained model or not. 
                Defaults to True.
            progress (bool, optional): Whether display progress bar of the download or not. 
                Defaults to False.
            no_grad (bool, optional): Whether paraemters in pretrained model (excluding fc, 
                i.e. output nn.Linear layer) is set to no gradients or not. 
                Defaults to False.
        
        Attributes:
            length (int): Size of embedding tensor.
            model (torchvision.models): Pretrained model in torchvision, which its fc layer 
                is change to a nn.Linear with output size = embedding size.
        
        :Reference:

        #. `Docs of torchvision.models <https://pytorch.org/docs/stable/torchvision/models.html>`_.

        """
        # refer to parent class
        super(PretrainedImageInputs, self).__init__()

        # bind length to embed_size
        self.length = embed_size

        # bind model to the pretrained model in torchvision
        self.model = getattr(torchvision.models, model_name)(pretrained=pretrained, progress=progress)
        
        # change fc output layer to be a nn.Linear with output size = embedding size
        in_size = self.model.fc.in_features
        self.model.fc = nn.Linear(in_size, embed_size)

        # set requires_grad be False if no_grad is True
        if no_grad:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
        
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of PretrainedImageInputs
        
        Args:
            inputs (T), shape = (B, C, H_{i}, W_{i}), dtype = torch.float: Tensor of images.
        
        Returns:
            T, shape = (B, 1, E): Output of PretrainedImageInputs.
        """
        # feed forward to pretrained module
        outputs = self.model(inputs.rename(None))
        outputs.names = ("B", "N", "E")
        
        return outputs
