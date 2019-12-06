import torch
import torch.nn as nn
from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor

class OuterProductNetworkLayer(nn.Module):
    r"""Layer class of Outer Product Network. 
    
    Outer Product Network is in Product based Neural Network :title:`Yanru Qu et at, 2016`[1],
    by calculating outer product between embedded tensors element-wisely and compressing by a 
    kernel to get cross features interactions.
    
    :Reference:

    #. `Yanru Qu et at, 2016. Product-based Neural Networks for User Response Prediction <https://arxiv.org/abs/1611.00144>`_.
    
    """
    @no_jit_experimental_by_namedtensor
    def __init__(self, 
                 embed_size  : int,
                 num_fields  : int,
                 kernel_type : str = "mat"):
        r"""Initialize OuterProductNetworkLayer
        
        Args:
            embed_size (int): Size of embedding tensor
            num_fields (int): Number of inputs' fields
            kernel_type (str, optional): Type of kernel to compress outer-product. 
                Defaults to "mat".
        
        Attributes:
            rowidx (T), dtype = torch.long: 1st indices to index inputs in 2nd dimension for inner product.
            colidx (T), dtype = torch.long: 2nd indices to index inputs in 2nd dimension for inner product.
            kernel (nn.Parameter): Parameter of kernel in outer product network.
            kernel_type (str): Type of kernel to compress outer-product.
        
        Raises:
            ValueError: when kernel_size is not in ["mat", "num", "vec"]
        """
        # Refer to parent class
        super(OuterProductNetworkLayer, self).__init__()

        # Create rowidx and colidx to index inputs for outer product
        self.rowidx = list()
        self.colidx = list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.rowidx.append(i)
                self.colidx.append(j)
        self.rowidx = torch.LongTensor(self.rowidx)
        self.colidx = torch.LongTensor(self.colidx)

        # Calculate kernel size to compress outer product's output
        if kernel_type == "mat":
            kernel_size = (embed_size, num_fields * (num_fields - 1) // 2, embed_size)
        elif kernel_type == "vec":
            kernel_size = (1, num_fields * (num_fields - 1) // 2, embed_size)
        elif kernel_type == "num":
            kernel_size = (1, num_fields * (num_fields - 1) // 2, 1)
        else:
            raise ValueError('kernel_type only allows: ["mat", "num", "vec"].')
        
        # Initialize kernel by xavier normal
        self.kernel = nn.Parameter(torch.zeros(kernel_size))
        nn.init.xavier_normal_(self.kernel.data)

        # Bind kernel_type to kernel_type
        self.kernel_type = kernel_type
    
    def extra_repr(self) -> str:
        r"""Return information in print-statement of layer.
        
        Returns:
            str: Information of print-statement of layer.
        """
        return 'kernel_type={}'.format(self.kernel_type)

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of OuterProductNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, NC2), dtype = torch.float: Output of OuterProductNetworkLayer.
        """
        # Indexing data for outer product
        # inputs: emb_inputs, shape = (B, N, E)
        # output: p, shape = (B, NC2, E)
        # output: q, shape = (B, NC2, E)
        p = emb_inputs.rename(None)[:, self.rowidx]
        q = emb_inputs.rename(None)[:, self.colidx]

        # Rename tensor names of p and q
        p.names = ("B", "N", "E")
        q.names = ("B", "N", "E")

        # Apply kernel on outer product
        if self.kernel_type == "mat":
            # Reshape p and apply kernel on it
            # inputs: p, shape = (B, NC2, E)
            # inputs: kernel, shape = (E, NC2, E)
            # output: kp, shape = (B, E, NC2, E)
            kp = p.unflatten("N", [("H", 1), ("N", p.size("N"))]) * self.kernel
            
            # Aggregate kp on last dimension, and reshape it
            # inputs: kp, shape = (B, E, NC2, E)
            # output: kp, shape = (B, NC2, E)
            kp = kp.sum(dim="E").align_to("B", "N", "H").rename(H="E")

            # Multiply q on kp and aggregate it on dimension E
            # inputs: kp, shape = (B, NC2, E)
            # inputs: q, shape = (B, NC2, E)
            # output: outputs, shape = (B, NC2)
            outputs = (kp * q).sum(dim="E")

        else:
            # Multiply q and kernel on p, and aggregate on dimension E
            # inputs: p, shape = (B, NC2, E)
            # inputs: q, shape = (B, NC2, E)
            # inputs: kernel, shape = (1, NC2, E or 1)
            # output: outputs, shape = (B, NC2)
            outputs = (p * q * self.kernel).sum(dim="E")

        # Rename tensor names
        outputs.names = ("B", "O")

        return outputs
