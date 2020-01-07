from . import _CtrModel
import torch
import torch.nn as nn
from torecsys.layers import DNNLayer, MOELayer
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
from typing import Callable, List

class DeepMixtureOfExpertsModel(_CtrModel):
    r"""Model class of Deep Mixture-of-Experts (MoE) model.

    Deep Mixture-of-Experts is purposed by David Eigen et at at 2013, which is to combine 
    outputs of several `expert` models, each of which specializes in a different part of input 
    space. To combine them, a gate, which is a stack of linear and softmax, will be trained 
    for weigthing outputs of expert before return.

    :Reference:

    #. `David Eigen et al, 2013. Learning Factored Representations in a Deep Mixture of Experts <https://arxiv.org/abs/1312.4314>`_.

    """
    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size       : int,
                 num_fields       : int,
                 num_experts      : int,
                 moe_layer_sizes  : List[int],
                 deep_layer_sizes : List[int],
                 deep_dropout_p   : List[float] = None,
                 deep_activation  : Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize DeepMixtureOfExpertsModel
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            num_experts (int): Number of experts' model
            moe_layer_sizes (List[int]): Size of mixture-of-experts models' outputs
            deep_layer_sizes (List[int]): Layer sizes of dense network in expert's model
            deep_dropout_p (List[float], optional): Probability of Dropout in dense network in expert's model. 
                Defaults to None.
            deep_activation (Callable[[T], T], optional): Activation function of dense network in expert's model. 
                Defaults to nn.ReLU().
        """
        # Refer to parent class
        super(DeepMixtureOfExpertsModel, self).__init__()

        # Calculate first input size and concatenate it with layer_sizes
        inputs_size = embed_size * num_fields
        layer_sizes = [inputs_size] + moe_layer_sizes

        # Initialize a list of mixture-of-experts layers
        self.moes = nn.ModuleList()

        for i, (inp, out) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            # Calculate input's size from last moe layer
            inp = num_experts * inp if i != 0 else inp
            moe = MOELayer(
                inputs_size = inp,
                output_size = num_experts * out,
                num_experts = num_experts,
                expert_func = DNNLayer,
                expert_inputs_size = inp,
                expert_output_size = out,
                expert_layer_sizes = deep_layer_sizes,
                expert_dropout_p   = deep_dropout_p,
                expert_activation  = deep_activation
            )
            self.moes.append(moe)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of DeepMixtureOfExpertsModel
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of DeepMixtureOfExpertsModel
        """
        # loop through mixture-of-experts layers
        for moe in self.moes:
            # calculate with mixture-of-experts layer forwardly
            # inputs: emb_inputs, shape = (B, N, E)
            # output: emb_inputs, shape = (B, N = 1, E = O)
            emb_inputs = moe(emb_inputs).rename(O="E")
        
        # Aggregate emb_inputs on dimension E = O
        # inputs: emb_inputs, shape = (B, N, E)
        # output: output, shape = (B, O = 1)
        output = emb_inputs.sum(dim="E").rename(N="O")
        
        # Drop names of outputs, since autograd doesn't support NamedTensor yet.
        output = output.rename(None)
        
        return output