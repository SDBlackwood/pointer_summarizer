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





