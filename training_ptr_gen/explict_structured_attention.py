"""
Explict Structural Attention Code

Influnced by latent structural attention code written here
https://github.com/vidhishanair/structured-text-representations/blob/master/models/modules/StructuredAttention.py


Returns:
    [type]: [description]
"""

import torch.nn as nn
import torch
class ExplictStructuredAttention(nn.Module):
    def __init__(self, hidden_dimension, config):
        super(ExplictStructuredAttention, self).__init__()
        self.hidden_dimension = hidden_dimension;
        self.use_cuda  = config.use_gpu and torch.cuda.is_available()

        """Linear layer for `u`
        # (256, 100)
        """
        self.F_u = nn.Linear(self.hidden_dimension * 2, 100, bias=True)
        torch.nn.init.xavier_uniform_(self.F_u.weight)
        nn.init.constant_(self.F_u.bias, 0)

        """Linear layer for `e`
        """
        self.F_e = nn.Linear(100, 100, bias=True)
        torch.nn.init.xavier_uniform_(self.F_e.weight)
        nn.init.constant_(self.F_e.bias, 0)

    def forward(self, input, structure): 
        """Forward function called at each step of the decoder

        Attributes
        ---------
        input: torch.Tensor - torch.Size([batch_size,structural_dimension_size,hidden_dimension_size])
        structure: torch.Tensor - tensor of torch.Size(batch_size,structural_dimension_size) e.g (8,100)
        """
        structure = torch.unsqueeze(structure, 2).expand(torch.Size([8,100,100]))
        # Set to cuda if we are using
        if self.use_cuda:
            structure=structure.to("cuda")
            input=input.to("cuda")

        u_i = torch.tanh(self.F_u(input))
        t_i = u_i * structure
        e_i = torch.tanh(self.F_e(t_i))

        return e_i