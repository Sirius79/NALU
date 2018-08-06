import torch
import torch.nn as nn
from torch import optim
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math

class NACCell(nn.Module):
  '''
    Neural Accumulator Cell
  '''
  def __init__(self, input_size, output_size):
    '''
      input_size: input dimension
      output_size: output dimension
    '''
    super(NACCell, self).__init__()
    self.input = input_size
    self.output = output_size
    
    self.W_h = Parameter(torch.Tensor(self.output, self.input, device=device))
    self.M_h = Parameter(torch.Tensor(self.output, self.input, device=device))
    self.W = Parameter(torch.tanh(self.W_h) * torch.sigmoid(self.M_h))
    self.register_parameter('bias', None)
    
    init.xavier_uniform_(self.W_h)
    init.xavier_uniform_(self.M_h)
  
  def forward(self, input):
    return F.linear(input, self.W, self.bias)
