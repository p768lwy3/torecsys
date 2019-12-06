from . import _CtrModel
from torecsys.layers import BilinearInteractionLayer, DNNLayer, SENETLayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from torecsys.utils.utils import combination
import torch
import torch.nn as nn
from typing import Callable, List


class FeatureImportanceAndBilinearFeatureInteractionNetwork(_CtrModel):
    r"""Model class of Feature-Importance and Bilinear-Feature-Interaction Network (FiBiNet).

    Feature-Importance and Bilinear-Feature-Interaction Network was proposed by Tongwen Huang 
    in Sina Weibo Inc. in 2019, which is:
    
    #. to implement a famous computer vision algorithm `SENET` on recommendation system.

    #. to apply bilinear calculation to calculate features interactions rather than using 
    inner-product or hadamard product, where they were used in recommendation system always.

    :Reference:

    #. `Tongwen Huang et al, 2019. FibiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction <https://arxiv.org/abs/1905.09433>`_.
    
    """
    def __init__(self, 
                 embed_size       : int,
                 num_fields       : int,
                 senet_reduction  : int,
                 deep_output_size : int,
                 deep_layer_sizes : List[int],
                 bilinear_type    : str = "all",
                 bilinear_bias    : bool = True,
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize FeatureImportanceAndBilinearFeatureInteractionNetwork
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            senet_reduction (int): Size of reduction in dense layer of senet.
            deep_output_size (int): Output size of dense network
            deep_layer_sizes (List[int]): Layer sizes of dense network
            bilinear_type (str, optional): Type of bilinear to calculate interactions.
                Defaults to "all".
            bilinear_bias (bool, optional): Flag to control using bias in bilinear-interactions.
                Defaults to True.
            deep_dropout_p (List[float], optional): Probability of Dropout in dense network.
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of dense network.
                Defaults to nn.ReLU().
        
        Attributes:
            senet (nn.Module): Module of Squeeze-and-Excitation Network layer.
            bilinear (nn.Module): Module of Bilinear-interaction layer.
            deep (nn.Module): Module of dense layer.
        """
        # Refer to parent class
        super(FeatureImportanceAndBilinearFeatureInteractionNetwork, self).__init__()

        # Initialize senet layer
        self.senet = SENETLayer(num_fields, senet_reduction)

        # Initialize bilinear interaction layer
        self.emb_bilinear = BilinearInteractionLayer(embed_size, num_fields, bilinear_type, bilinear_bias)
        self.senet_bilinear = BilinearInteractionLayer(embed_size, num_fields, bilinear_type, bilinear_bias)

        # Calculate inputs' size of DNNLayer, i.e. output's size of ffm (= NC2) * embed_size * 2
        inputs_size = combination(num_fields, 2)
        inputs_size = inputs_size * embed_size * 2

        # Initialize dense layer
        self.deep = DNNLayer(
            inputs_size = inputs_size,
            output_size = deep_output_size,
            layer_sizes = deep_layer_sizes,
            dropout_p   = deep_dropout_p,
            activation  = deep_activation
        )
    
    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of FeatureImportanceAndBilinearFeatureInteractionNetwork
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of FeatureImportanceAndBilinearFeatureInteractionNetwork
        """

        # Calculate senet-like embedding by senet
        # inputs: emb_inputs, shape = (B, N, E)
        # output: senet_emb, shape = (B, N, E)
        senet_emb = self.senet(emb_inputs.rename(None))

        # Calculate bilinear-interaction of emb_inputs
        # inputs: emb_inputs, shape = (B, N, E)
        # output: emb_interation, shape = (B, NC2, E)
        emb_interation = self.emb_bilinear(emb_inputs.rename(None))
        emb_interation.names = ("B", "N", "E")

        # Calculate bilinear-interaction of senet_emb
        # inputs: senet_emb, shape = (B, N, E)
        # output: senet_interaction, shape = (B, NC2, E)
        senet_interaction = self.senet_bilinear(senet_interaction.rename(None))
        senet_interaction.names = ("B", "N", "E")

        # Concatenate emb_interation and senet_interaction and flatten the output into 2-dimension
        # inputs: emb_interation, shape = (B, NC2, E)
        # inputs: senet_interaction, shape = (B, NC2, E)
        # output: output, shape = (B, O = E * NC2 * 2)
        output = torch.cat([emb_interation, senet_interaction], dim="N")
        output = output.flatten(["N", "E"], "O")

        # Calculate forwardly with dense layer
        # inputs: output, shape = (B, O = E * NC2 * 2)
        # output: output, shape = (B, O = 1)
        output = self.deep(output.rename(None))

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)

        return output
