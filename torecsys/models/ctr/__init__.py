r"""torecsys.models.ctr is a sub module of the implementation of the whole models of CTR prediction
"""

from .. import _Model

class _CtrModel(_Model):
    def __init__(self):
        super(_CtrModel, self).__init__()

from .attentional_factorization_machine import AttentionalFactorizationMachineModel
from .deep_and_cross_network import DeepAndCrossNetworkModel
from .deep_ffm import DeepFieldAwareFactorizationMachineModel
from .deep_fm import DeepFactorizationMachineModel
from .deep_mcp import DeepMatchingCorrelationPredictionModel
from .elaborated_entire_space_supervised_multi_task import ElaboratedEntireSpaceSupervisedMultiTaskModel
from .entire_space_multi_task import EntireSpaceMultiTaskModel
from .factorization_machine import FactorizationMachineModel
from .factorization_machine_supported_neural_network import FactorizationMachineSupportedNeuralNetworkModel
from .fat_deep_ffm import FieldAttentiveDeepFieldAwareFactorizationMachineModel
from .field_aware_factorization_machine import FieldAwareFactorizationMachineModel
from .logistic_regression import LogisticRegressionModel
from .neural_collaborative_filtering import NeuralCollaborativeFilteringModel

# create shorten name of models
AFMModel = AttentionalFactorizationMachineModel
DeepFFMModel = DeepFieldAwareFactorizationMachineModel
DeepFMModel = DeepFactorizationMachineModel
FATDeepFFMModel = FieldAttentiveDeepFieldAwareFactorizationMachineModel
FieldAwareNeuralFactorizationMachineModel = DeepFieldAwareFactorizationMachineModel
FFMModel = FieldAwareFactorizationMachineModel
FMModel = FactorizationMachineModel
FMNNModel = FactorizationMachineSupportedNeuralNetworkModel
FNFMModel = FieldAwareNeuralFactorizationMachineModel
NCFModel = NeuralCollaborativeFilteringModel
