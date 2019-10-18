from torecsys.utils.decorator import jit_experimental
import torch
import torch.nn as nn

class WideLayer(nn.Module):
    r"""Layer class of wide layer, which is a stack of linear and dropout, used in calculation 
    of linear relation frequently.
    """
    @jit_experimental
    def __init__(self,
                 output_size : int,
                 embed_size  : int = None,
                 num_fields  : int = None,
                 inputs_size : int = None,
                 dropout_p   : float = 0.0):
        r"""Initialize WideLayer
        
        Args:
            output_size (int): Output size of wide layer
            embed_size (int, optional): Size of embedding tensor. 
                Required with num_fields. 
                Defaults to None.
            num_fields (int, optional): Number of inputs' fields. 
                Required with embed_size together. 
                Defaults to None.
            inputs_size (int, optional): Size of inputs. 
                Required when embed_size and num_fields are None. 
                Defaults to None.
            dropout_p (float, optional): Probability of Dropout in wide layer. 
                Defaults to 0.0.
        
        Attributes:
            inputs_size (int): Size of inputs, or Product of embed_size and num_fields.
            model (torch.nn.Sequential): Sequential of wide layer.
     
        Raises:
            ValueError: when embed_size or num_fields is missing if using embed_size and num_field pairs, or when inputs_size is missing if using inputs_size
        """
        # refer to parent class
        super(WideLayer, self).__init__()

        # set inputs_size to N * E when using embed_size and num_fields
        if inputs_size is None and embed_size is not None and num_fields is not None:
            inputs_size = embed_size * num_fields
        # else, set inputs_size to inputs_size
        elif inputs_size is not None and (embed_size is None or num_fields is None):
            inputs_size = inputs_size
        else:
            raise ValueError("Only allowed:\n    1. embed_size and num_fields is not None, and inputs_size is None\n    2. inputs_size is not None, and embed_size or num_fields is None")
        
        # bind inputs_size to inputs_size
        self.inputs_size = inputs_size

        # initialize sequential of model
        self.model = nn.Sequential()

        # add modules including linear and dropout to model
        self.model.add_module("linear_0", nn.Linear(inputs_size, output_size))
        self.model.add_module("dropout_0", nn.Dropout(dropout_p))
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of WideLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of wide layer.
        """
        # reshape inputs from (B, N, E) to (B, N * E) 
        # or from (B, 1, I) to (B, I)
        ## emb_inputs = emb_inputs.view(-1, self.inputs_size)
        emb_inputs = emb_inputs.flatten(["N", "E"], "E")

        # forward to model and return output with shape = (B, O)
        outputs = self.model(emb_inputs.rename(None))

        # .unsqueeze(1) to transform the shape into (B, 1, O) before return
        ## outputs = outputs.unsqueeze(1)
        outputs.names = ("B", "O")
        return outputs
    