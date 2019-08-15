r"""torecsys.models.ctr.modules is a sub module of the implementation of modules in severals models, which can be concatenate to CTR model
"""

from torecsys.modules import _Module

class _CtrModule(_Module):
    def __init__(self):
        super(_CtrModule, self).__init__()

from .factorization_machine import FactorizationMachineModule
# from field_aware_factorization_machine import FieldAwareFactorizationMachineModule
