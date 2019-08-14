import torch
import torch.nn as nn
from texttable import Texttable
from typing import List, Tuple


class Regularization(nn.Module):
    r"""Regularization is a Module to calculate regularized loss of a model
    """
    def __init__(self,
                 model        : nn.Module,
                 weight_decay : float = 0.01,
                 norm         : int   = 2):
        r"""initialize Regularization to calculate p-th norm regularized loss of model
        
        Args:
            model (nn.Module): nn.Module of model to be calculated regularized loss
            weight_decay (float, optional): multiplier of regularized loss. Defaults to 0.01.
            norm (int, optional): order of norm to calculate regularized loss. Defaults to 2.
        
        Raises:
            ValueError: when weight_decay smaller than or equal to 0.0
        """
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            raise ValueError("weight_decay must be greater than 0.0.")

        self.model = model
        self.weight_decay = weight_decay
        self.norm = norm
        
        self.weights_list = self.get_weights(model)
        self.weights_info()
    
    def get_weights(self, model: nn.Module) -> List[Tuple[str, nn.Parameter]]:
        r"""Get weights by iterated with model.named_parameters()
        
        Args:
            model (nn.Module): model for regularization
        
        Returns:
            List[Tuple[str, nn.Parameter]]: list of tuple with name and weights' parameters
        """
        weights_list = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weights_list.append((name, param))
        return weights_list

    def weights_info(self):
        r"""print weights' name to be regularized in Texttable
        """
        t = Texttable()
        t.add_rows([["Weight Name: "]] + \
                   [[name.replace(".weight", "")] for name, weight in self.weights_list])
        print(t)

    def regularize_loss(self, weights_list: List[Tuple[str, nn.Parameter]]) -> torch.Tensor:
        r"""function to calculate regularized loss
        
        Args:
            weights_list (List[Tuple[str, nn.Parameter]]): list of tuple of names and paramters to calculate the regularized loss
        
        Returns:
            torch.Tensor, shape = (1, ), dtype = torch.float: regularized loss
        """
        loss = 0.0
        for name, weight in weights_list:
            reg = torch.norm(weight, p=self.norm)
            loss += reg

        return self.weight_decay * loss
    
    def forward(self, model: nn.Module) -> torch.Tensor:
        r"""calculate regularized loss
        
        Args:
            model (nn.Module): nn.Module of model to be calculated regularized loss
        
        Returns:
            torch.Tensor: regularized loss
        """
        self.weights_list = self.get_weights(model)
        reg_loss = self.regularize_loss(self.weights_list)
        return reg_loss
