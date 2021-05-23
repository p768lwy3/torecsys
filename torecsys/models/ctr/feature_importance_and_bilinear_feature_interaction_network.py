from typing import List, Optional

import torch
import torch.nn as nn

from torecsys.layers import BilinearInteractionLayer, DNNLayer, SENETLayer
from torecsys.models.ctr import CtrBaseModel
from torecsys.utils.operations import combination


class FeatureImportanceAndBilinearFeatureInteractionNetwork(CtrBaseModel):
    """
    Model class of Feature-Importance and Bilinear-Feature-Interaction Network (FiBiNet).

    Feature-Importance and Bilinear-Feature-Interaction Network was proposed by Tongwen Huang 
    in Sina Weibo Inc. in 2019, which is:
    
    #. to implement a famous computer vision algorithm `SENET` on recommendation system.

    #. to apply bilinear calculation to calculate features interactions rather than using 
    inner-product or hadamard product, where they were used in recommendation system always.

    :Reference:

    #. `Tongwen Huang et al, 2019. FibiNET: Combining Feature Importance and Bilinear feature Interaction for
        Click-Through Rate Prediction <https://arxiv.org/abs/1905.09433>`_.
    
    """

    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 senet_reduction: int,
                 deep_output_size: int,
                 deep_layer_sizes: List[int],
                 bilinear_type: Optional[str] = 'all',
                 bilinear_bias: Optional[bool] = True,
                 deep_dropout_p: Optional[List[float]] = None,
                 deep_activation: Optional[nn.Module] = nn.ReLU()):
        """
        Initialize FeatureImportanceAndBilinearFeatureInteractionNetwork
        
        Args:
            embed_size (int): size of embedding tensor
            num_fields (int): number of inputs' fields
            senet_reduction (int): size of reduction in dense layer of senet.
            deep_output_size (int): output size of dense network
            deep_layer_sizes (List[int]): layer sizes of dense network
            bilinear_type (str, optional): type of bilinear to calculate interactions. Defaults to "all"
            bilinear_bias (bool, optional): flag to control using bias in bilinear-interactions. Defaults to True
            deep_dropout_p (List[float], optional): probability of Dropout in dense network. Defaults to None
            deep_activation (torch.nn.Module, optional): activation function of dense network. Defaults to nn.ReLU()
        """
        super().__init__()

        inputs_size = combination(num_fields, 2)
        inputs_size = inputs_size * embed_size * 2

        self.senet = SENETLayer(num_fields, senet_reduction, squared=False)
        self.emb_bilinear = BilinearInteractionLayer(embed_size, num_fields, bilinear_type, bilinear_bias)
        self.senet_bilinear = BilinearInteractionLayer(embed_size, num_fields, bilinear_type, bilinear_bias)
        self.deep = DNNLayer(
            inputs_size=inputs_size,
            output_size=deep_output_size,
            layer_sizes=deep_layer_sizes,
            dropout_p=deep_dropout_p,
            activation=deep_activation
        )

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of FeatureImportanceAndBilinearFeatureInteractionNetwork
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, O), data_type = torch.float: output of FeatureImportanceAndBilinearFeatureInteractionNetwork
        """
        # Calculate bilinear-interaction of emb_inputs
        # inputs: emb_inputs, shape = (B, N, E)
        # output: emb_interaction, shape = (B, NC2, E)
        emb_interaction = self.emb_bilinear(emb_inputs.rename(None))
        emb_interaction.names = ('B', 'N', 'E',)

        # Calculate senet-like embedding by senet
        # inputs: emb_inputs, shape = (B, N, E)
        # output: senet_emb, shape = (B, N, E)
        senet_emb = self.senet(emb_inputs.rename(None))

        # Calculate bilinear-interaction of senet_emb
        # inputs: senet_emb, shape = (B, N, E)
        # output: senet_interaction, shape = (B, NC2, E)
        senet_interaction = self.senet_bilinear(senet_emb.rename(None))
        senet_interaction.names = ('B', 'N', 'E',)

        # Concatenate emb_interaction and senet_interaction and flatten the output into 2-dimension
        # inputs: emb_interaction, shape = (B, NC2, E)
        # inputs: senet_interaction, shape = (B, NC2, E)
        # output: outputs, shape = (B, O = E * NC2 * 2)
        outputs = torch.cat([emb_interaction, senet_interaction], dim='N')
        outputs = outputs.flatten(('N', 'E',), 'O')

        # Calculate forwardly with dense layer
        # inputs: output, shape = (B, O = E * NC2 * 2)
        # output: output, shape = (B, O = 1)
        outputs = self.deep(outputs.rename(None))

        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        outputs = outputs.rename(None)
        return outputs
