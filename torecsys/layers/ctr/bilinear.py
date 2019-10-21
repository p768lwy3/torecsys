from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn


class BilinearNetworkLayer(nn.Module):
    r"""Layer class of Bilinear to calculate interation in element-wise by nn.Bilinear, which the 
    calculation is: for i-th layer, :math:`x_{i} = (x_{0} * A_{i} * x_{i - 1}) + b_{i} + x_{0}`, 
    where :math:`A_{i}` is the weight of module of shape :math:`(O_{i}, I_{i1}, I_{i2})`.
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 num_layers  : int,
                 embed_size  : int = None, 
                 num_fields  : int = None,
                 inputs_size : int = None):
        r"""Initialize BilinearNetworkLayer

        Args:
            num_layers (int): Number of layers of Bilinear Network
            embed_size (int, optional): Size of embedding tensor. 
                Required with num_fields. 
                Defaults to None.
            num_fields (int, optional): Number of inputs' fields. 
                Required with embed_size together. 
                Defaults to None.
            inputs_size (int, optional): Size of inputs. 
                Required when embed_size and num_fields are None. 
                Defaults to None.
        
        Attributes:
            inputs_size (int): Size of inputs, or Product of embed_size and num_fields.
            model (torch.nn.ModuleList): Module List of Bilinear Layers.
        
        Raises:
            ValueError: when embed_size or num_fields is missing if using embed_size and 
                num_field pairs, or when inputs_size is missing if using inputs_size
        """
        # refer to parent class
        super(BilinearNetworkLayer, self).__init__()

        # set inputs_size to N * E when using embed_size and num_fields
        if inputs_size is None and embed_size is not None and num_fields is not None:
            inputs_size = embed_size * num_fields
        # else, set inputs_size to inputs_size
        elif inputs_size is not None and (embed_size is None or num_fields is None):
            inputs_size = inputs_size
        else:
            raise ValueError("Only allowed:\n    1. embed_size and num_fields is not None, " + 
                "and inputs_size is None.\n    2. inputs_size is not None, and embed_size " + 
                "or num_fields is None.")
        
        # bind inputs_size to inputs_size
        self.inputs_size = inputs_size

        # initialize module list for Bilinear
        self.model = nn.ModuleList()

        # add modules to module list of Bilinear
        for _ in range(num_layers):
            self.model.append(nn.Bilinear(inputs_size, inputs_size, inputs_size))
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """Forward calculation of BilinearNetworkLayer
        
        Args:
            emb_inputs (T), shape = shape = (B, N, E) or (B, 1, I), dtype = torch.float: Embedded features tensors.
        
            Returns:
                T, shape = (B, 1, N * E) or (B, 1, I), dtype = torch.float: Output of BilinearNetworkLayer.
        """
        # reshape inputs from (B, N, E) to (B, 1, N * E) if inputs' shape is (B, N, E)
        # or from (B, 1, I) to (B, I)
        ## emb_inputs = emb_inputs.view(-1, self.inputs_size)
        emb_inputs = emb_inputs.flatten(["N", "E"], "O")

        # copy emb_inputs to outputs for residual
        outputs = emb_inputs.detach().requires_grad_()

        # forward calculation of bilinear and add residual
        for layer in self.model:
            # return size = (B, N * E) or (B, I)
            ## outputs = layer(emb_inputs, outputs) + emb_inputs
            outputs = layer(emb_inputs.rename(None), outputs.rename(None))
            outputs.names = ("B", "O")
            outputs = outputs + emb_inputs
        
        # .unsqueeze(1) to transform the shape into (B, 1, O) before return
        ## outputs = outputs.unsqueeze(1)

        return outputs
