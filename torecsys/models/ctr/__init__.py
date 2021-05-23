"""
torecsys.models.ctr is a sub model of the implementation of the whole models of CTR prediction
"""

from torecsys.models import BaseModel


class CtrBaseModel(BaseModel):
    def __init__(self):
        super().__init__()


from torecsys.models.ctr.attentional_factorization_machine import AttentionalFactorizationMachineModel
from torecsys.models.ctr.deep_and_cross_network import DeepAndCrossNetworkModel
from torecsys.models.ctr.deep_ffm import DeepFieldAwareFactorizationMachineModel
from torecsys.models.ctr.deep_fm import DeepFactorizationMachineModel
from torecsys.models.ctr.deep_mcp import DeepMatchingCorrelationPredictionModel
from torecsys.models.ctr.deep_session_interest_network import DeepSessionInterestNetworkModel
from torecsys.models.ctr.deep_moe import DeepMixtureOfExpertsModel
from torecsys.models.ctr.elaborated_entire_space_supervised_multi_task import \
    ElaboratedEntireSpaceSupervisedMultiTaskModel
from torecsys.models.ctr.entire_space_multi_task import EntireSpaceMultiTaskModel
from torecsys.models.ctr.factorization_machine import FactorizationMachineModel
from torecsys.models.ctr.factorization_machine_supported_neural_network import \
    FactorizationMachineSupportedNeuralNetworkModel
from torecsys.models.ctr.fat_deep_ffm import FieldAttentiveDeepFieldAwareFactorizationMachineModel
from torecsys.models.ctr.feature_importance_and_bilinear_feature_interaction_network import \
    FeatureImportanceAndBilinearFeatureInteractionNetwork
from torecsys.models.ctr.field_aware_factorization_machine import FieldAwareFactorizationMachineModel
from torecsys.models.ctr.logistic_regression import LogisticRegressionModel
from torecsys.models.ctr.multigate_moe import MultiGateMixtureOfExpertsModel
from torecsys.models.ctr.neural_collaborative_filtering import NeuralCollaborativeFilteringModel
from torecsys.models.ctr.neural_factorization_machine import NeuralFactorizationMachineModel
from torecsys.models.ctr.position_bias_aware_learning_framework import PositionBiasAwareLearningFrameworkModel
from torecsys.models.ctr.product_neural_network import ProductNeuralNetworkModel
from torecsys.models.ctr.wide_and_deep import WideAndDeepModel
from torecsys.models.ctr.xdeep_fm import XDeepFactorizationMachineModel

AFM = AttentionalFactorizationMachineModel
DeepFFM = DeepFieldAwareFactorizationMachineModel
DeepFM = DeepFactorizationMachineModel
DSIN = DeepSessionInterestNetworkModel
FATDeepFFM = FieldAttentiveDeepFieldAwareFactorizationMachineModel
FieldAwareNeuralFactorizationMachine = DeepFieldAwareFactorizationMachineModel
FFM = FieldAwareFactorizationMachineModel
FM = FactorizationMachineModel
FMNN = FactorizationMachineSupportedNeuralNetworkModel
FNFM = FieldAwareNeuralFactorizationMachine
NCF = NeuralCollaborativeFilteringModel
NFM = NeuralFactorizationMachineModel
PAL = PositionBiasAwareLearningFrameworkModel
PNN = ProductNeuralNetworkModel
xDeepFM = XDeepFactorizationMachineModel
