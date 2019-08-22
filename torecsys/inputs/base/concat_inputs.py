from . import _Inputs
import torch
from typing import List


class ConcatInputs(nn.Module):
    r"""
    """
    def __init__(self, schema: List[tuple]):
        r"""[summary]
        
        Args:
            nn ([type]): [description]
            schema (List[tuple]): [description]
        """
        super(ConcatInputs, self).__init__()

        # store the schema to self
        self.schema = schema
    
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        r"""[summary]
        
        Args:
            inputs (Dict[str, torch.Tensor]): [description]
        
        Returns:
            torch.Tensor: [description]
        """
        outputs = list()

        for args_tuple in self.schema:
            # get basic args
            embedding = args_tuple[0]
            inp_name = args_tuple[1]
            
            inp_val = [inputs[i] for i in inp_name]
            inp_val = torch.cat(inp_val, dim=1)
            args = [inp_val]
            
            if embedding.__class__.__name__ == "SequenceIndexEmbedding":
                arg_name = args_tuple[2][0]
                args.append(inputs[arg_name])
            
            outputs.append(embedding(*args))

        outputs = torch.cat(outputs, dim=2)
        
        return outputs
        