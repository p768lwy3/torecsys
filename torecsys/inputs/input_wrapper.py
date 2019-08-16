from . import _Inputs
from typing import Dict, Tuple

class InputsWrapper(_Inputs):
    def __init__(self, 
                 schema: Dict[str, Tuple[str, _Inputs]]):
        r"""InputsWrapper is a wrapper to concatenate numbers of _Inputs of different fields
        
        Args:
            schema (Dict[str, Tuple[str, _Inputs]]): schema of wrapper
        
        Key-Values:
            output name: tuple of schema, where the 1st element is input name and the 2nd element is the type of _Inputs in torecsys.inputs.base
        """
        super(InputsWrapper, self).__init__()
