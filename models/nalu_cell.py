import torch
import torch.nn as nn
from torch import optim
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math
from nac_cell import NACCell

class NeuralALUCell(nn.Module):
  '''
    Neural Arithmetic Logic Unit Cell 
  '''
  def __init__(self, input_size, output_size, epsilon = 1e-5):
    '''
      input_size: input dimension
      output_size: output dimension
      epsilon: prevents log0
    '''
    super(NeuralALUCell, self).__init__()
    self.input = input_size
    self.output = output_size
    self.epsilon = epsilon
    
    self.G = Parameter(torch.Tensor(self.output, self.input, device=device))
    self.W = Parameter(torch.Tensor(self.output, self.input, device=device))
    self.register_parameter('bias', None)
    self.nac = NACCell(self.input, self.output)
    
    init.xavier_uniform_(self.G)
    init.xavier_uniform_(self.W)
  
  def forward(self, input):
    '''
      g*a: addition-subtraction
      (1-g)*m: multiplication-division
    '''
    a = self.nac(input)
    g = F.sigmoid(F.linear(input, self.G, self.bias))
    m = torch.exp(F.linear(torch.log(torch.abs(input) + self.epsilon)), self.W, self.bias)
    return g*a + (1-g)*m
