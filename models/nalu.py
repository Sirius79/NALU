import torch
import torch.nn as nn
from torch import optim
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import math
from nalu_cell import NeuralALUCell

class NALU(nn.Module):
  '''
    stacked NALU Cells
  '''
  def __init__(self, input_size, hidden_size, output_size, num_layers, epsilon = 1e-5):
    '''
      input_size: input dim
      hidden_size: hidden dim
      output_size: output dim
      num_layers: number of layers
      epsilon: prevents log0
    '''
    super(NALU, self).__init__()
    self.input = input_size
    self.hidden = hidden_size
    self.output = output_size
    self.num_layers = num_layers
    self.epsilon = epsilon
    
    layers = [NeuralALUCell(self.hidden if i>0 else self.input, self.hidden if i<num_layers-1 else self.output, self.epsilon) for i in num_layers]
    self.model = nn.Sequential(*layers)
  
  def forward(self, input):
    return self.model(input)
