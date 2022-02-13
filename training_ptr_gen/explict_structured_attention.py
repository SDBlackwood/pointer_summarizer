"""
This code has been directly used from the below sourc.  
All credit for this code goes to the authors.  
https://github.com/vidhishanair/structured-text-representations/blob/master/models/modules/StructuredAttention.py

Example 
self.sentence_structure_att = StructuredAttention(device, self.sem_dim_size, self.sent_hidden_size, bidirectional, py_version)
self.document_structure_att = StructuredAttention(device, self.sem_dim_size, self.doc_hidden_size, bidirectional, py_version)


Returns:
    [type]: [description]
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import math 
#from data_util.utils import calc_mem, format_mem

class ExplictStructuredAttention(nn.Module):
    def __init__(self, hidden_dimension, config):
        super(ExplictStructuredAttention, self).__init__()
        
        self.hidden_dimension = hidden_dimension;

        self.use_cuda  = config.use_gpu and torch.cuda.is_available()

        # (256, 256)
        self.F_u = nn.Linear(self.hidden_dimension * 2, self.hidden_dimension* 2, bias=True)
        torch.nn.init.xavier_uniform_(self.F_u.weight)
        nn.init.constant_(self.F_u.bias, 0)

        # (256, 256)
        self.F_e = nn.Linear(self.hidden_dimension * 2, self.hidden_dimension* 2, bias=True)
        torch.nn.init.xavier_uniform_(self.F_e.weight)
        nn.init.constant_(self.F_e.bias, 0)

    def forward(self, input, structure): 


        structure = torch.unsqueeze(structure, 2).expand(input.size())
        if self.use_cuda:
            structure=structure.to("cuda")
            input=input.to("cuda")

        u_i = torch.tanh(self.F_u(input))
        t_i = u_i * structure
        e_i = torch.tanh(self.F_u(t_i))

        return e_i