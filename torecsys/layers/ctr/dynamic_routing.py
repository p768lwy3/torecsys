import numpy as np
import torch
import torch.nn as nn
from torecsys.utils.decorator import no_jit_experimental_by_namedtensor
from torecsys.utils.operations import squash

class DynamicRoutingLayer(nn.Module):
    r"""Layer class of Behaviour to Interest Dynamic Routing (B2I Dynamic Routing).

    Behaviour to Interest Dynamic Routing is purposed by :title:`Chao Li et al, 2019`[1] to 
    transform users' behaviour to users' interest by a variant of Capsule Neural Network, 
    which is a new architecture purposed by :title:`Sara Sabour et al, 2017`[2] for image 
    recognition to solve the issues due to backpropogation by taking vectors' inputs and 
    generate vectors' outputs and keep more information of inference by length anf angle. 

    Behaviour to Interest Dynamic Routing make two changes comparing with the origin: 

    #. Instead of using K projection matrices (i.e. project by different matrix for each 
    activity capsules), all activity capsules share single projection matrices to solve an 
    isses due to the length difference between difference user-item interactions.

    #. To solve an issuse due to the above change, use Gaussian Initializer for the 
    projection matrix instead of initializing by zero to prevent the same outputs for each 
    activity capsule.

    #. Instead of calculating a fixed number of capsule j (marked by K), the number of 
    activity capcules is calculated dynamically by the following formula: 
    :math:`K'_{u} = max(1, min(K, log_{2}(\left | I_{u} \right |)))`.

    :Reference:

    #. `Chao Li et al, 2019. Multi-Interest Network with Dynamic Routing for Recommendation at Tmall<https://arxiv.org/abs/1904.08030>`_.

    #. `Sara Sabour, 2017 et al. Dynamic Routing Between Capsules <https://arxiv.org/abs/1710.09829>`_.

    """
    def __init__(self,
                 embed_size  : int, 
                 routed_size : int,
                 num_caps    : int,
                 num_iters   : int):
        r"""Initialize DynamicRoutingLayer
        
        Args:
            embed_size (int): Size of embedding tensor
            routed_size (int): Size of routed tensor, i.e. output size.
            num_caps (int): Maximum number of capsules.
            num_iters (int): Number of iterations to update coupling coefficients.
        
        Attributes:
            num_caps (int): Maximum number of capsules.
            num_iters (int): Number of iterations to update coupling coefficients.
            S (nn.Parameter): Parameter to project capsule i to capsule j.
        """
        # Refer to parent class
        super(DynamicRoutingLayer, self).__init__()

        # Bind num_iters to num_iters
        self.num_caps = num_caps
        self.num_iters = num_iters

        # Initialize parameter S, which is shared by all interest capsules to 
        # project behaviour tensors e_i to interest tensors \hat{e}_{j|i}.
        self.S = nn.Parameter(torch.randn(embed_size, routed_size))
        self.S.names = ("E", "Cout")

    def _dynamic_interest_number(self, I: int) -> int:
        r"""Calculate number of interest capsules adaptively
        
        Args:
            I (int): Number of items in items set interacted with a given user.
        
        Returns:
            int: Number of interest capsules.
        """
        return int(max(1, min(self.num_caps, np.log2(I))))

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of DynamicRoutingLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, N = N_cap, O = Ecap), dtype = torch.float: Output of DynamicRoutingLayer
        """
        # Calculate number of intetest capsules K
        # inputs: emb_inputs, shape = (B, N, E)
        # output: num_caps, int
        num_caps = self._dynamic_interest_number(emb_inputs.size("N"))
        
        # Calculate priors = \hat(e)_{j|i} for each capsule
        # inputs: emb_inputs, shape = (B, N, E_i)
        # inputs: num_caps, int
        # output: priors, shape = (B, K, N, Ecap)
        batch_size = emb_inputs.size("B")
        priors = torch.matmul(emb_inputs, self.S)
        priors = priors.unflatten("B", [("B", batch_size), ("C", 1)])
        priors = priors.rename(None).repeat(1, num_caps, 1, 1)
        priors.names = ("B", "K", "N", "Ecap")
        
        # Detach priors as priors_temp to prevent gradients from flowing
        # inputs: priors, shape = (B, K, N, Ecap)
        # output: priors_temp, shape = (B, K, N, Ecap)
        priors_temp = priors.detach()
        
        # Initialize coupling coefficient by bij ∼ N(0, σ^2).
        # inputs: priors_temp, shape = (B, K, N, Ecap)
        # output: coup_coeff, shape = (B, K, N, Ecap)
        coup_coeff = torch.randn_like(priors_temp.rename(None), device=priors.device)
        coup_coeff.names = priors_temp.names
        
        # Update coupling coefficient by iterative dynamic routing process
        for i in range(self.num_iters - 1):
            # Take softmax along num_caps to calculate weights for behaviour capsule.
            # inputs: coup_coeff, shape = (B, K, N, Ecap)
            # output: weights, shape = (B, K, N, Ecap)
            weights = torch.softmax(coup_coeff, dim="K")
            
            # calculate z
            # inputs: weights, shape = (B, K, N, Ecap)
            # inputs: u_hat, shape = (B, K, N, Ecap)
            # output: z, shape = (B, K, Ecap)
            z = (weights * priors_temp).sum(dim="N")
            
            # apply squashing non-linearity to z 
            # inputs: z, shape = (B, K, Ecap)
            # output: v, shape = (B, K, Ecap)
            v = squash(z)
            
            # calculate dot product between v and \hat{u]_{j|i}
            # inputs: priors_temp, shape = (B, K, N, Ecap)
            # inputs: v, shape = (B, K, Ecap)
            # output: uv, shape = (B, K, N, Ecap = 1)
            v_temp = v.unflatten("Ecap", [("Ecap", v.size("Ecap")), ("N", 1)])
            similarity = torch.matmul(priors_temp.rename(None), v_temp.rename(None))
            similarity.names = coup_coeff.names
            
            # update bij for all behavior capsule i and interest capsule j
            # inputs: coup_coeff, shape = (B, K, N, Ecap)
            # inputs: similarity, shape = (B, K, N, Ecap)
            # output: coup_coeff, shape = (B, K, N, Ecap)
            coup_coeff = coup_coeff + similarity
        
        # calculate output with the original u_hat without routing updates
        # inputs: priors, shape = (B, K, N, Ecap)
        # inputs: coup_coeff, shape = (B, K, N, Ecap)
        # output: output, shape = (B, K', E)
        weights = torch.softmax(coup_coeff, dim="K")
        z = (weights * priors).sum(dim="N")
        
        # apply squashing non-linearity to z 
        # inputs: z, shape = (B, K, Ecap)
        # output: output, shape = (B, K, Ecap)
        output = squash(z)

        # rename output names to (B, N, O)
        output.names = ("B", "N", "O")
        
        return output
