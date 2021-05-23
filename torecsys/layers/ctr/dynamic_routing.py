from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from torecsys.layers import BaseLayer
from torecsys.utils.operations import squash


class DynamicRoutingLayer(BaseLayer):
    r"""
    Layer class of Behaviour to Interest Dynamic Routing (B2I Dynamic Routing).

    Behaviour to Interest Dynamic Routing is purposed by :title:`Chao Li et al, 2019`[1] to 
    transform users' behaviour to users' interest by a variant of Capsule Neural Network, 
    which is a new architecture purposed by :title:`Sara Sabour et al, 2017`[2] for image 
    recognition to solve the issues due to back propagation by taking vectors' inputs and
    generate vectors' outputs and keep more information of inference by length anf angle. 

    Behaviour to Interest Dynamic Routing make two changes comparing with the origin: 

    #. Instead of using K projection matrices (i.e. project by different matrix for each 
    activity capsules), all activity capsules share single projection matrices to solve an 
    issues due to the length difference between difference user-item interactions.

    #. To solve an issue due to the above change, use Gaussian Initializer for the
    projection matrix instead of initializing by zero to prevent the same outputs for each 
    activity capsule.

    #. Instead of calculating a fixed number of capsule j (marked by K), the number of 
    activity capsules is calculated dynamically by the following formula:
    :math:`K'_{u} = max(1, min(K, log_{2}(\left | I_{u} \right |)))`.

    :Reference:

    #. `Chao Li et al, 2019. Multi-Interest Network with Dynamic Routing for Recommendation at
    Tmall<https://arxiv.org/abs/1904.08030>`_.

    #. `Sara Sabour, 2017 et al. Dynamic Routing Between Capsules <https://arxiv.org/abs/1710.09829>`_.

    """

    @property
    def inputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'N', 'E',)
        }

    @property
    def outputs_size(self) -> Dict[str, Tuple[str, ...]]:
        return {
            'inputs': ('B', 'Number of Caps', 'Routed Size',)
        }

    def __init__(self,
                 embed_size: int,
                 routed_size: int,
                 max_num_caps: int,
                 num_iter: int):
        """
        Initialize DynamicRoutingLayer
        
        Args:
            embed_size (int): size of embedding tensor
            routed_size (int): size of routed tensor, i.e. output size
            max_num_caps (int): maximum number of capsules
            num_iter (int): number of iterations to update coupling coefficients
        """
        super().__init__()

        self.max_num_caps = max_num_caps
        self.num_caps = None
        self.num_iter = num_iter

        self.S = nn.Parameter(torch.randn(embed_size, routed_size))
        self.S.names = ('E', 'COut')

    def _dynamic_interest_number(self, i: int) -> int:
        """
        Calculate number of interest capsules adaptively
        
        Args:
            i (int): number of items in items set interacted with a given user
        
        Returns:
            int: number of interest capsules
        """
        return int(max(1, min(self.max_num_caps, np.log2(i))))

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward calculation of DynamicRoutingLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), data_type = torch.float: embedded features tensors
        
        Returns:
            T, shape = (B, N = N_cap, O = ECap), data_type = torch.float: output of DynamicRoutingLayer
        """
        # Name the inputs tensor for alignment
        emb_inputs.names = ('B', 'N', 'E',)

        # calculate number of interest capsules K
        # inputs: emb_inputs, shape = (B, N, E)
        # output: max_num_caps, int
        self.num_caps = self._dynamic_interest_number(emb_inputs.size('N'))

        # calculate priors = \hat(e)_{j|i} for each capsule
        # inputs: emb_inputs, shape = (B, N, E_i)
        # inputs: max_num_caps, int
        # output: priors, shape = (B, K, N, ECap)
        batch_size = emb_inputs.size('B')
        priors = torch.matmul(emb_inputs, self.S)
        priors = priors.unflatten('B', (('B', batch_size,), ('C', 1,),))
        priors = priors.rename(None).repeat(1, self.num_caps, 1, 1)
        priors.names = ('B', 'K', 'N', 'ECap')

        # detach priors as priors_temp to prevent gradients from flowing
        # inputs: priors, shape = (B, K, N, ECap)
        # output: priors_temp, shape = (B, K, N, ECap)
        priors_temp = priors.detach()

        # initialize coupling coefficient by bij ∼ N(0, σ^2).
        # inputs: priors_temp, shape = (B, K, N, ECap)
        # output: coup_coefficient, shape = (B, K, N, ECap)
        coup_coefficient = torch.randn_like(priors_temp.rename(None), device=priors.device)
        coup_coefficient.names = priors_temp.names

        # update coupling coefficient by iterative dynamic routing process
        for _ in range(self.num_iter - 1):
            # take softmax along max_num_caps to calculate weights for behaviour capsule.
            # inputs: coup_coefficient, shape = (B, K, N, ECap)
            # output: weights, shape = (B, K, N, ECap)
            weights = torch.softmax(coup_coefficient, dim='K')

            # calculate z
            # inputs: weights, shape = (B, K, N, ECap)
            # inputs: u_hat, shape = (B, K, N, ECap)
            # output: z, shape = (B, K, ECap)
            z = (weights * priors_temp).sum(dim='N')

            # apply squashing non-linearity to z 
            # inputs: z, shape = (B, K, ECap)
            # output: v, shape = (B, K, ECap)
            v = squash(z)

            # calculate dot product between v and \hat{u]_{j|i}
            # inputs: priors_temp, shape = (B, K, N, ECap)
            # inputs: v, shape = (B, K, ECap)
            # output: uv, shape = (B, K, N, ECap = 1)
            v_temp = v.unflatten('ECap', (('ECap', v.size('ECap'),), ('N', 1,),))
            similarity = torch.matmul(priors_temp.rename(None), v_temp.rename(None))
            similarity.names = coup_coefficient.names

            # update bij for all behavior capsule i and interest capsule j
            # inputs: coup_coefficient, shape = (B, K, N, ECap)
            # inputs: similarity, shape = (B, K, N, ECap)
            # output: coup_coefficient, shape = (B, K, N, ECap)
            coup_coefficient = coup_coefficient + similarity

        # calculate output with the original u_hat without routing updates
        # inputs: priors, shape = (B, K, N, ECap)
        # inputs: coup_coefficient, shape = (B, K, N, ECap)
        # output: output, shape = (B, K', E)
        weights = torch.softmax(coup_coefficient, dim='K')
        z = (weights * priors).sum(dim='N')

        # apply squashing non-linearity to z 
        # inputs: z, shape = (B, K, ECap)
        # output: output, shape = (B, K, ECap)
        output = squash(z)

        # rename output names to (B, N, O)
        output.names = ('B', 'N', 'O',)

        return output
