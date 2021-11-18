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

class StructuredAttention(nn.Module):
    def __init__(self, device, sem_dim_size, sent_hiddent_size, bidirectional, py_version):
        super(StructuredAttention, self).__init__()
        self.device = device
        self.bidirectional = bidirectional
        self.sem_dim_size = sem_dim_size
        self.str_dim_size = sent_hiddent_size - self.sem_dim_size
        self.pytorch_version = py_version
        print("Setting pytorch "+self.pytorch_version+" version for Structured Attention")

        self.tp_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tp_linear.weight)
        nn.init.constant_(self.tp_linear.bias, 0)

        self.tc_linear = nn.Linear(self.str_dim_size, self.str_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.tc_linear.weight)
        nn.init.constant_(self.tc_linear.bias, 0)

        self.fi_linear = nn.Linear(self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.fi_linear.weight)

        self.bilinear = nn.Bilinear(self.str_dim_size, self.str_dim_size, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.bilinear.weight)

        self.exparam = nn.Parameter(torch.Tensor(1,1,self.sem_dim_size))
        torch.nn.init.xavier_uniform_(self.exparam)

        self.fzlinear = nn.Linear(3*self.sem_dim_size, self.sem_dim_size, bias=True)
        torch.nn.init.xavier_uniform_(self.fzlinear.weight)
        nn.init.constant_(self.fzlinear.bias, 0)

    def forward(self, input): #batch*sent * token * hidden
        batch_size, token_size, dim_size = input.size()

        # Decompose the encoder output vector in two parts
        if(self.bidirectional):
            # input [8, 400, 512] -> [8, 400, 2, 256]
            input = input.view(batch_size, token_size, 2, dim_size//2)
            # semantic_vector [8, 400, 462]
            semantic_vector = torch.cat((input[:,:,0,:self.sem_dim_size//2],input[:,:,1,:self.sem_dim_size//2]),2)
            # semantic_vector [8, 400, 50]
            structural_vector = torch.cat((input[:,:,0,self.sem_dim_size//2:],input[:,:,1,self.sem_dim_size//2:]),2)
        else:
            semantic_vector = input[:,:,:self.sem_dim_size]
            structural_vector = input[:,:,self.sem_dim_size:]

        tp = torch.tanh(self.tp_linear(structural_vector)) # b*s, token, h1
        tc = torch.tanh(self.tc_linear(structural_vector)) # b*s, token, h1

        # [8, 400, 462] -> [8, 400, 400, 462] 
        tp = tp.unsqueeze(2).expand(
                tp.size(0), 
                tp.size(1), 
                tp.size(1), 
                tp.size(2)
        ).contiguous()
        # [8, 400, 462] -> [8, 400, 400, 462] 
        tc = tc.unsqueeze(2).expand(
            tc.size(0), 
            tc.size(1),
            tc.size(1), 
            tc.size(2)
        ).contiguous()
        # Unnormalised attention scores between `i` and `j`
        # [8,400,400]
        f_ij = self.bilinear(tp, tc).view( # b*s, token , token
            batch_size, 
            token_size, 
            token_size
        )
        # [8,400]
        f_i = torch.exp(self.fi_linear(structural_vector)).view(
                batch_size, 
                token_size
        )  

        # [8,400,400]
        mask = torch.ones(f_ij.size(1), f_ij.size(1)) - torch.eye(f_ij.size(1), f_ij.size(1))
        mask = mask.unsqueeze(0).expand(f_ij.size(0), mask.size(0), mask.size(1)).to(self.device)
        # [8,400,400]
        # Attenion Vector scores between tokens `i` and `j`
        A_ij = torch.exp(f_ij)*mask


        tmp = torch.sum(A_ij, dim=1)
        res = torch.zeros(batch_size, token_size, token_size).to(self.device)

        res.as_strided(tmp.size(), [res.stride(0), res.size(2) + 1]).copy_(tmp)
        
        # L is the Laplacian matrix
        L_ij = -A_ij + res   #A_ij has 0s as diagonals
        L_ij_bar = L_ij
        L_ij_bar[:,0,:] = f_i

        #No batch inverse
        LLinv = torch.stack([torch.inverse(li) for li in L_ij_bar])

        d0 = f_i * LLinv[:,:,0]

        #(1−δ1,j)Aij[ L−1]jj−(1−δi,1)Aij[ ̄L−1]ji
        LLinv_diag = torch.diagonal(LLinv, dim1=-2, dim2=-1).unsqueeze(2)

        tmp1 = (A_ij.transpose(1,2) * LLinv_diag ).transpose(1,2)
        tmp2 = A_ij * LLinv.transpose(1,2)

        temp11 = torch.zeros(batch_size, token_size, 1)
        temp21 = torch.zeros(batch_size, 1, token_size)

        temp12 = torch.ones(batch_size, token_size, token_size-1)
        temp22 = torch.ones(batch_size, token_size-1, token_size)

        mask1 = torch.cat([temp11,temp12],2).to(self.device)
        mask2 = torch.cat([temp21,temp22],1).to(self.device)

        dx = mask1 * tmp1 - mask2 * tmp2

        d = torch.cat([d0.unsqueeze(1), dx], dim = 1)
        df = d.transpose(1,2)

        ssr = torch.cat([self.exparam.repeat(batch_size,1,1), semantic_vector], 1)
        # Conxtext Vector gathered from possible parents of ui and ci
        pi = torch.bmm(df, ssr)
        # Conxtext Vector gathered from possible children
        ci = torch.bmm(dx, semantic_vector)

        # `ri` is a structure infused hidden representation for each encoder timestep `i`
        # ri=tanh(Wr[ei,pi,ci]
        ri = F.relu(
            self.fzlinear(
                    torch.cat(
                        [semantic_vector, pi, ci],
                        dim = 2
                    )
            )
        )

        """ 
        Reshape the model. Shape out of the Encoder is:
        - [batch_size, max_encoder_size, 2 * hidden_dim]
        - [8, 400, 512]
        Shape out of the structural attention network is:

        - [8, 400, 50]
        Why is the structural netwwork cutting the 3rd dimension down? 
        """
        # output = output.contiguous().view(
        #     batch_size, 
        #     sent_size, 
        #     token_size, 
        #     output.size(2)
        # )

        return ri, df

def b_inv(b_mat, device):
    eye = torch.rand(b_mat.size(0), b_mat.size(1), b_mat.size(2)).to(device)
    b_inv, _ = torch.gesv(eye, b_mat)
    return b_inv