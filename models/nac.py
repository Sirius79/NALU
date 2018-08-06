import torch
import torch.nn as nn
from torch import optim
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from nac_cell import NACCell

class NAC(nn.Module):
  '''
    stack Neural Accumulator Cells
  '''
  def __init__(self, input_size, hidden_size, output_size, num_layers):
    '''
      input_size: input dim
      hidden_size: hidden dim
      output_size: output dim
      num_layers: number of layers
    '''
    super(NAC, self).__init__()
    self.input = input_size
    self.hidden = hidden_size
    self.output = output_size
    self.num_layers = num_layers
    
    layers = [NACCell(self.hidden if i>0 else self.input, self.hidden if i<num_layers-1 else self.output) for i in num_layers]
    self.model = nn.Sequential(*layers)
  
  def forward(self, input):
    return self.model(input)
