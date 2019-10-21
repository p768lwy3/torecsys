from torecsys.utils.decorator import jit_experimental, no_jit_experimental_by_namedtensor
import torch
import torch.nn as nn


class OuterProductNetworkLayer(nn.Module):
    r"""Layer class of Outer Product Network used in Product based Neural Network :title:`Yanru Qu et at, 2016`[1], 
    by calculating outer product between embedded tensors element-wisely and compressing by a kernel to get cross features interactions
    
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
            kernel (torch.nn.Parameter): Parameter of kernel in outer product network.
            kernel_type (str): Type of kernel to compress outer-product.
        
        Raises:
            ValueError: when kernel_size is not in ["mat", "num", "vec"]
        """
        # refer to parent class
        super(OuterProductNetworkLayer, self).__init__()

        # create rowidx and colidx to index inputs for outer product
        self.rowidx = list()
        self.colidx = list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                self.rowidx.append(i)
                self.colidx.append(j)
        self.rowidx = torch.LongTensor(self.rowidx)
        self.colidx = torch.LongTensor(self.colidx)

        # set kernel size of kernel to compress outer product's output
        if kernel_type == "mat":
            kernel_size = (embed_size, num_fields * (num_fields - 1) // 2, embed_size)
        elif kernel_type == "vec":
            kernel_size = (1, num_fields * (num_fields - 1) // 2, embed_size)
        elif kernel_type == "num":
            kernel_size = (1, num_fields * (num_fields - 1) // 2, 1)
        else:
            raise ValueError('kernel_type only allows: ["mat", "num", "vec"].')
        
        # initialize kernel with xavier_normal_
        self.kernel = nn.Parameter(torch.zeros(kernel_size))
        nn.init.xavier_normal_(self.kernel.data)

        # bind kernel_type to kernel_type
        self.kernel_type = kernel_type
    
    def extra_repr(self):
        return 'kernel_type={}'.format(
            self.kernel_type
        )

    def forward(self, emb_inputs: torch.Tensor) -> torch.Tensor:
        r"""Forward calculation of OuterProductNetworkLayer
        
        Args:
            emb_inputs (T), shape = (B, N, E), dtype = torch.float: Embedded features tensors.
        
        Returns:
            T, shape = (B, NC2), dtype = torch.float: Output of OuterProductNetworkLayer.
        """
        # indexing data for outer product
        p = emb_inputs.rename(None)[:, self.rowidx] # shape = (B, NC2, E)
        q = emb_inputs.rename(None)[:, self.colidx] # shape = (B, NC2, E)

        # set name to p and q
        p.names = ("B", "N", "E")
        q.names = ("B", "N", "E")

        # apply kernel on outer product
        if self.kernel_type == "mat":
            # unsqueeze p to (B, 1, NC2, E), 
            # then multiply kernel and return shape = (B, E, NC2, E) 
            ## kp = p.unsqueeze(1) * self.kernel
            kp = p.unflatten("N", [("H", 1), ("N", p.size("N"))]) * self.kernel
            
            # aggregate last dimension of kp and return shape = (B, E, NC2)
            # then tranpose to shape = (B, NC2, E)
            ## kp = kp.sum(dim=-1).transpose(1, 2)
            kp = kp.sum(dim="E").align_to("B", "N", "H").rename(H="E")

            # multiply q to kp and return shape = (B, NC2, E)
            # then aggregate outputs with last dimension to shape (B, NC2)
            ## outputs = (kp * q).sum(dim=-1)
            outputs = (kp * q).sum(dim="E")

        else:
            # multiply q and kernel to p and return shape = (B, NC2, E)
            # then aggregate outputs with last dimension to shape (B, NC2)
            ## outputs = (p * q * self.kernel).sum(dim=-1)
            outputs = (p * q * self.kernel).sum(dim="E")
        
        # .unsqueeze(1) to transform the shape into (B, 1, O) before return
        ## outputs.unsqueeze(1)

        outputs.names = ("B", "O")
        return outputs
