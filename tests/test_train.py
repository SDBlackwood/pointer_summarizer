import unittest
import torch.nn as nn
import torch

import sys
sys.path.append(".")

from data_util import config
from training_ptr_gen.train import Train

class TrainTestCase(unittest.TestCase):

   def test_train_batch(self):
          
      # Train
      train_processor = Train()
      train_processor.trainIters(1, None)

   def test_linear(self):
          
         m = nn.Linear(2, 1)
         input = torch.randn(2, 2)
         output = m(input)
         print(output.size())

