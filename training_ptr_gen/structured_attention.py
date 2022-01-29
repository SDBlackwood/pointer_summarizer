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

class StructuredAttention(nn.Module):
    def __init__(self, device, sem_dim_size, sent_hiddent_size, bidirectional, py_version):
        super(StructuredAttention, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.sem_dim_size = sem_dim_size
        self.str_dim_size = sent_hiddent_size - self.sem_dim_size
        self.pytorch_version = py_version
        print("Setting pytorch "+self.pytorch_version+" version for Structured Attention")

        # Store a memmory array to caclulate the total memory for the linear layers
        mem = []

        # (412, 412)
        self.W_p = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.W_p.weight)
        nn.init.constant_(self.W_p.bias, 0)
        #mem.append(calc_mem(self.W_p))

        # (412, 412)
        self.W_c = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.W_c.weight)
        nn.init.constant_(self.W_c.bias, 0)
        #mem.append(calc_mem(self.W_c))

        # (412, 412)
        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fi_linear.weight)
        #mem.append(calc_mem(self.fi_linear))

        self.W_a = nn.Bilinear(self.str_dim_size, self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.W_a.weight)
        #mem.append(calc_mem(self.W_a))

        self.exparam = nn.Parameter(torch.Tensor(1,1,self.sem_dim_size))
        torch.nn.init.xavier_uniform_(self.exparam)
        self.W_r = nn.Linear(3*self.sem_dim_size, self.sem_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.W_r.weight)
        nn.init.constant_(self.W_r.bias, 0)
        #mem.append(calc_mem(self.W_r))

        #format_mem(sum(mem))


    def forward(self, input): 
        batch_size, token_size, dim_size = input.size()

        # Decompose the encoder ouu_jut vector in two parts
        """
        We feed our input tokens (wi) to a bi-LSTM encoder to obtain hidden state 
        representations `hi` . We decompose the hidden state vector into two parts:
        `di` and `ei`, which we call the structural part and the semantic part respectively
        """
        # input [8, 400, 512] -> [8, 400, 2, 256]
        input = input.view(batch_size, token_size, 2, dim_size//2)
        # e_i [8, 400, 462]
        e_i = torch.cat((input[:,:,0,:self.sem_dim_size//2],input[:,:,1,:self.sem_dim_size//2]),2)
        # d_i [8, 400, 50]
        d_i = torch.cat((input[:,:,0,self.sem_dim_size//2:],input[:,:,1,self.sem_dim_size//2:]),2)
        """
        For  every  pair  of  two  input  tokens,  
        we  transform  their structural  parts `d` 
        and  try  to  compute  the  probability  
        of  a parent-child  relationship  edge  between  them  
        in  the  dependency tree
        """
        u_j = torch.tanh(self.W_p(d_i)) # b*s, token, h1
        u_k = torch.tanh(self.W_c(d_i)) # b*s, token, h1

        """
        Restructure the intertoken matrix into a 3D tensor
        """
        # [8, 400, 462] -> [8, 400, 400, 462] 
        u_j = u_j.unsqueeze(2).expand(
                u_j.size(0), 
                u_j.size(1), 
                u_j.size(1), 
                u_j.size(2)
        ).contiguous()
        # [8, 400, 462] -> [8, 400, 400, 462] 
        u_k = u_k.unsqueeze(2).expand(
            u_k.size(0), 
            u_k.size(1),
            u_k.size(1), 
            u_k.size(2)
        ).contiguous()
        """
        Next, we compute an intertoken attention function `f_jk` 
        where `W_a` is also learned
        """
        # Unnormalised attention scores between `j` and `k`
        # [8,400,400]
        f_jk = self.W_a(u_j, u_k).view( # b*s, token , token
            batch_size, 
            token_size, 
            token_size
        )
        del u_j
        del u_k
        # [8,400]
        f_i = torch.exp(self.fi_linear(d_i)).view(
                batch_size, 
                token_size
        )  
        del d_i

        # [8,400,400]
        mask = f_jk.new_ones(f_jk.size(1), f_jk.size(1)) - torch.eye(f_jk.size(1), f_jk.size(1))
        mask = mask.unsqueeze(0).expand(f_jk.size(0), mask.size(0), mask.size(1)).to(self.device)
        
        """
        For a document with `K` tokens, `f` is a `KxK` matrix representing
        intertoken attention.  We model each token as a node in the dependency 
        tree and define the probability of an edge between tokens at positions
        `j` and `k`, P(zjk= 1), which is given as:
        - A_jk = { 0 if j=k, else exp(f_ik))
        """
        # [8,400,400]
        # Attenion Vector scores between tokens `i` and `j`
        A_jk = torch.exp(f_jk)*mask
        del mask

        tmp = torch.sum(A_jk, dim=1)
        res = torch.zeros(batch_size, token_size, token_size).to(self.device)
        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        del tmp

        
        """
        We use this (soft) dependency tree formulation to compute a structural 
        representation `r` for each encoder token as
        """
        # L is the Laplacian matrix
        L_ij = -A_jk + res   #A_jk has 0s as diagonals
        L_ij_bar = L_ij
        L_ij_bar[:,0,:] = f_i
        del res
        del L_ij

        #No bau_kh inverse
        LLinv = torch.stack([torch.inverse(li) for li in L_ij_bar])
        del L_ij_bar


        d0 = f_i * LLinv[:,:,0]
        del f_i

        #(1−δ1,j)Aij[ L−1]jj−(1−δi,1)Aij[ ̄L−1]ji
        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_jk.transpose(1,2) * LLinv_diag ).transpose(1,2)
        tmp2 = A_jk * LLinv.transpose(1,2)
        del LLinv_diag
        del LLinv

        temp11 = A_jk.new_zeros(batch_size, token_size, 1)
        temp21 = A_jk.new_zeros(batch_size, 1, token_size)

        temp12 = A_jk.new_ones(batch_size, token_size, token_size-1)
        temp22 = A_jk.new_ones(batch_size, token_size-1, token_size)
        del A_jk

        mask1 = torch.cat([temp11,temp12],2).to(self.device)
        mask2 = torch.cat([temp21,temp22],1).to(self.device)

        """
        We use this (soft) dependency tree formulation to compute a structural 
        representation `r` for each encoder token as
        """
        a_ik = mask1 * tmp1 - mask2 * tmp2
        a_ki = torch.cat([d0.unsqueeze(1), a_ik], dim = 1).transpose(1,2)
        del mask1
        del mask2

        ssr = torch.cat([self.exparam.repeat(batch_size,1,1), e_i], 1)
        # Conxtext Vector gathered from possible parents of ui and ci
        si = torch.bmm(a_ki, ssr)
        # Conxtext Vector gathered from possible children
        ci = torch.bmm(a_ik, e_i)
        r_i = torch.tanh(
            self.W_r(
                    torch.cat(
                        [e_i, si, ci],
                        dim = 2
                    )
            )
        )
        del e_i, a_ik, ssr, si, ci

        return r_i, a_ki