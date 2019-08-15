r"""torecsys.models.ctr.modules is a sub module of the implementation of modules in severals models, which can be concatenate to CTR model
"""

from torecsys.modules import _Module

class _CtrModule(_Module):
    def __init__(self):
        super(_CtrModule, self).__init__()

from .attentional_factorization_machine import AttentionalFactorizationMachineModule
from .deep_and_cross_network import DeepAndCrossNetworkModule
from .deep_ffm import DeepFieldAwareFactorizationMachineModule

from .factorization_machine import FactorizationMachineModule
# from field_aware_factorization_machine import FieldAwareFactorizationMachineModule
