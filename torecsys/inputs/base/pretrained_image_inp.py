from . import _Inputs
import torch
import torch.nn as nn
import torchvision


class PretrainedImageInputs(_Inputs):
    r"""PretrainedImageInputs is a input field to compute features vectors of images by pre-trained CV-model
    """
    def __init__(self,
                 embed_size : int,
                 model_name : str,
                 pretrained : bool = True,
                 progress   : bool = False,
                 no_grad    : bool = False):
        r"""initialize the pretrained image inputs
        
        Args:
            embed_size (int): embedding size
            model_name (str): string of model name. Refer to: `torchvision.models <https://pytorch.org/docs/stable/torchvision/models.html>`_.
            pretrained (bool, optional): boolean flag of torchvision.models to use pre-trained model. Defaults to True.
            progress (bool, optional): boolean flag of torchvision.models to display progress bar of the download. Defaults to False.
            no_grad (bool, optional): boolean flag to set the requires_grad of paraemters in model, excluding fc, i.e. output nn.Linear layer. Defaults to False.
        """
        super(PretrainedImageInputs, self).__init__()

        self.length = embed_size
        self.model = getattr(torchvision.models, model_name)(pretrained=pretrained, progress=progress)
        
        # change fc output layer to be a nn.Linear, where output size = embedding size
        in_size = self.mode.fc.in_features
        self.model.fc = nn.Linear(in_size, embed_size)

        # set requires_grad be False if no_grad is True
        if no_grad:
            for name, param in self.model.named_parameters():
                if not name.startswith("fc"):
                    param.requires_grad = False
        
    
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        r"""Return features vectors of inputs calculated by pre-trained CV-model
        
        Args:
            inputs (torch.Tensor), shape = (batch size, number of channels, image height, image width), dtype = torch.float: image tensor
        
        Returns:
            torch.Tensor, shape = (batch size, 1, embedding size): features vectors
        """
        outputs = self.model(inputs)
        return outputs
