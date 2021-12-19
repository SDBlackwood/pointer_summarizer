import sys
sys.path.append(".")
from training_ptr_gen.train import Train
from data_util import config
import unittest
import torch.nn as nn
import torch



class TrainTestCase(unittest.TestCase):

   def test_train_batch(self):

      # Train
      train_processor = Train()
      train_processor.trainIters(1, None)

   def test_memory(self):

      linear = nn.Bilinear(50, 512, 1)

      # Get the shape
      result = ((linear.in_features * linear.out_features) * (4 / (1024^3)))*2
      print(result)
      train_processor.trainIters(1, None)

   def test_size(self):

      self.encoder_size = 256
      self.batch_size = 8
      self.F_u = nn.Linear(self.encoder_size , self.encoder_size, bias=True)
      torch.nn.init.xavier_uniform_(self.F_u.weight)
      nn.init.constant_(self.F_u.bias, 0)

      # (200, 200)
      self.F_e = nn.Linear(self.encoder_size , self.encoder_size, bias=True)
      torch.nn.init.xavier_uniform_(self.F_e.weight)
      nn.init.constant_(self.F_e.bias, 0)

      input = torch.rand([8,200,256])
      structure = torch.rand([8,200,256])

      u_i = torch.tanh(self.F_u(input))
      t_i = u_i * structure
      e_i = torch.tanh(self.F_u(t_i))


   def test_max_pool(self):

      answer_index = torch.zeros([2,4], dtype=torch.int64)
      answer_index[:,2:4] += 1
      answer_index[1,:2] += 2
      answer_index[1,2:4] += 2
      # 
      encoder_outputs = torch.tensor([
         [1,1,2,2],
         [3,3,4,5]
      ])
      # 4 and 8
      results = []
      # For each row
      for x in range(answer_index.size(0)):
         result = []
         for item in torch.unique(answer_index[x,:]):
            # Create a tensor of just the values which are this unique
            result.append(torch.masked_select(encoder_outputs, (answer_index == item)).sum().item())
         results.append(result)
      answer_pool = torch.tensor(results)

      a = answer_pool
      # [10, 14, 1] where each value is the sum of the encoder values per quest

      # [v1,v2,v3] where each value is the vote score for each answer



def retrieve_elements_from_indices(tensor, indices):
   flattened_tensor = tensor.flatten(start_dim=2)
   output = flattened_tensor.gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
   return output

   def test_linear(self):

      ri = torch.rand(8,400,50)
      W_r = nn.Linear(50, 512)

      s_t_hat = torch.rand(8,512)
      v = nn.Linear(50, 1)
      W_s = nn.Linear(512, 50)
      b = 0
      encoder_outputs = torch.rand(8,400,512)
      batch, token, dim = list(encoder_outputs.size())

      Wrri = W_r(ri) # 8,400,512
      Wsst = W_s(s_t_hat) # 8,512
      Wsst = Wsst.unsqueeze(1)
      Wsst = Wsst.expand(8,400,50)
      Wsst = Wsst.contiguous().view(-1, 50)

      features =  Wrri + Wsst
      e_stuct_t_i = v(
         torch.tanh(
           features
         )
      )
      a_t_struct = torch.softmax(e_stuct_t_i, dim=2)
      # Calculate the structural context vector
      c_t_stuct = torch.bmm(a_t_struct, encoder_outputs)
      print(c_t_stuct)





