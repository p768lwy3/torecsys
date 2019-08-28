r"""torecsys.layers.ctr is a sub module of implementation of layers in click through rate prediction.
"""

from .attentional_factorization_machine import AttentionalFactorizationMachineLayer
from .bilinear import BilinearNetworkLayer
from .compose_excitation_network import ComposeExcitationNetworkLayer
from .compress_interaction_network import CompressInteractionNetworkLayer
from .cross_network import CrossNetworkLayer
from .factorization_machine import FactorizationMachineLayer
from .field_aware_factorization_machine import FieldAwareFactorizationMachineLayer
from .inner_product_network import InnerProductNetworkLayer
from .multilayer_perceptron import MultilayerPerceptronLayer
from .outer_product_network import OuterProductNetworkLayer
from .wide import WideLayer

# create shorten name of layers
AFMLayer = AttentionalFactorizationMachineLayer
CENLayer = ComposeExcitationNetworkLayer
CINLayer = CompressInteractionNetworkLayer
DenseLayer = MultilayerPerceptronLayer
DNNLayer = MultilayerPerceptronLayer
FFMLayer = FieldAwareFactorizationMachineLayer
FMLayer = FactorizationMachineLayer
FullyConnectLayer = MultilayerPerceptronLayer
FeedForwardLayer = MultilayerPerceptronLayer
