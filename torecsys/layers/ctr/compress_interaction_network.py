from typing import Callable, List

import torch
import torch.nn as nn

from torecsys.utils.decorator import no_jit_experimental_by_namedtensor


class CompressInteractionNetworkLayer(nn.Module):
    r"""Layer class of Compress Interaction Network (CIN).
    
    Compress Interaction Network was used in xDeepFM by Jianxun Lian et al, 2018.
    
    It compress cross-features tensors calculated by element-wise cross features interactions 
    with outer product by 1D convolution with a :math:`1 * 1` kernel.

    :Reference:

    #. `Jianxun Lian et al, 2018. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommendation
    Systems <https://arxiv.org/abs/1803.05170.pdf>`_.

    """

    @no_jit_experimental_by_namedtensor
    def __init__(self,
                 embed_size: int,
                 num_fields: int,
                 output_size: int,
                 layer_sizes: List[int],
                 is_direct: bool = False,
                 use_bias: bool = True,
                 use_batchnorm: bool = True,
                 activation: Callable[[torch.Tensor], torch.Tensor] = nn.ReLU()):
        r"""Initialize CompressInteractionNetworkLayer
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            output_size (int): Output size of compress interaction network
            layer_sizes (List[int]): Layer sizes of compress interaction network
            is_direct (bool, optional): Whether outputs is passed to next step directly or not.
                Defaults to False.
            use_bias (bool, optional): Whether bias added to Conv1d or not. 
                Defaults to True.
            use_batchnorm (bool, optional): Whether batch normalization is applied or not after Conv1d. 
                Defaults to True.
            activation (Callable[[T], T], optional): Activation function of Conv1d. 
                Defaults to nn.ReLU().
        
        Attributes:
            embed_size (int): Size of embedding tensor.
            layer_sizes (List[int]): List of integer concatenated by num_fields and layer_sizes.
            is_direct (bool): Flag to show outputs is passed to next step directly or not.
            model (torch.nn.ModuleList): Module List of compress interaction network.
            fc (torch.nn.Module): Fully-connect layer (i.e. Linear layer) of outputs.
        """
        # refer to parent class
        super(CompressInteractionNetworkLayer, self).__init__()

        # bind embed_size, is_direct to embed_size, is_direct
        self.embed_size = embed_size
        self.is_direct = is_direct

        # generate a list of num_fields and layer_sizes
        self.layer_sizes = [num_fields] + layer_sizes

        # initialize module list of model
        self.model = nn.ModuleList()

        # initialize cin sequential modules, and add them to module
        for i, (s_i, s_j) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            # create a module of sequential
            cin = nn.Sequential()

            # calculate channel size of Conv1d
            in_c = self.layer_sizes[0] * s_i
            out_c = s_j if is_direct or i == (len(self.layer_sizes) - 1) else s_j * 2

            # Initialize conv1d and add it to sequential module
            cin.add_module("Conv1d", nn.Conv1d(in_c, out_c, kernel_size=1, bias=use_bias))

            # Initialize batchnorm and add it to sequential module
            if use_batchnorm:
                cin.add_module("Batchnorm", nn.BatchNorm1d(out_c))

            # Initialize activation and add it to sequential module
            if activation is not None:
                cin.add_module("Activation", activation)

            # Add cin sequential to the module list
            self.model.append(cin)

        # Calculate output size of model
        model_output_size = int(sum(layer_sizes))

        # Initialize linear layer
        self.fc = nn.Linear(model_output_size, output_size)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of CompressInteractionNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, O), dtype = torch.float: Output of CompressInteractionNetworkLayer.
        """
        # Initialize two lists to store tensors of outputs and next steps temporarily
        direct_list = list()
        hidden_list = list()

        # Transpose emb_inputs
        # inputs: emb_inputs, shape = (B, N, E)
        # output: x0, shape = (B, E, N)
        x0 = emb_inputs.align_to("B", "E", "N")
        hidden_list.append(x0)

        # Expand dimension N of x0 to Nx (= N) and H (= 1)
        # inputs: x0, shape = (B, E, N)
        # output: x0, shape = (B, E, Nx = N, H = 1)
        x0 = x0.unflatten("N", [("Nx", x0.size("N")), ("H", 1)])

        # Calculate with cin forwardly
        for i, layer_size in enumerate(self.layer_sizes[:-1]):
            # Get tensors of previous step and reshape it
            # inputs: hidden_list[-1], shape = (B, E, N)
            # output: xi, shape = (B, E, H = 1, Ny = N)
            xi = hidden_list[-1]
            xi = xi.unflatten("N", [("H", 1), ("Ny", xi.size("N"))])

            # Calculate outer product of x0 and xi
            # inputs: x0, shape = (B, E, Nx = N, H = 1)
            # inputs: xi, shape = (B, E, H = 1, Ny = N) 
            # output: out_prod, shape = (B, E, Nx = N, Ny = N)
            ## out_prod = torch.matmul(x0, xi)
            out_prod = torch.einsum("ijkn,ijnh->ijkh", [x0.rename(None), xi.rename(None)])
            out_prod.names = ("B", "E", "Nx", "Ny")

            # Reshape out_prod
            # inputs: out_prod, shape = (B, E, Nx = N, Ny = N)
            # output: out_prod, shape = (B, N = Nx * Ny, E)
            out_prod = out_prod.flatten(["Nx", "Ny"], "N")
            out_prod = out_prod.align_to("B", "N", "E")

            # Apply convolution, batchnorm and activation
            # inputs: out_prod, shape = (B, N = Nx * Ny, E)
            # output: outputs, shape = (B, N = (Hi * 2 or Hi), E)
            outputs = self.model[i](out_prod.rename(None))
            outputs.names = ("B", "N", "E")

            if self.is_direct:
                # Pass to output directly
                # inputs: outputs, shape = (B, N = Hi, E)
                # output: direct, shape = (B, N = Hi, E)
                direct = outputs

                # Reshape and pass to next step directly
                # inputs: outputs, shape = (B, Hi, E)
                # output: hidden, shape = (B, E, N = Hi)
                hidden = outputs.align_to("B", "E", "N")
            else:
                if i != (len(self.layer_sizes) - 1):
                    # Split outputs into two part and pass them to outputs and hidden separately
                    # inputs: outputs, shape = (B, Hi * 2, E)
                    # output: direct, shape = (B, N = Hi, E)
                    # output: hidden, shape = (B, N = Hi, E)
                    direct, hidden = torch.chunk(outputs, 2, dim=1)

                    # Reshape and pass to next step
                    # inputs: hidden, shape = (B, N = Hi, E)
                    # output: hidden, shape = (B, E, N = Hi)
                    hidden = hidden.align_to("B", "E", "N")
                else:
                    # Pass to output directly
                    # inputs: outputs, shape = (B, N = Hi, E)
                    # output: direct, shape = (B, N = Hi, E)
                    direct = outputs
                    hidden = 0

            # Store tensors to lists temporarily
            direct_list.append(direct)
            hidden_list.append(hidden)

        # Concatenate direct_list into a tensor
        # inputs: direct_list, shape = (B, Hi, E)
        # output: outputs, shape = (B, sum(Hi), E)
        outputs = torch.cat(direct_list, dim="N")

        # Aggregate outputs on dimension E and pass to dense layer
        # inputs: outputs, shape = (B, sum(Hi), E)
        # output: outputs, shape = (B, O)
        outputs = self.fc(outputs.sum("E"))
        outputs.names = ("B", "O")

        return outputs
