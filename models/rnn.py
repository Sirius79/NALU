import torch
import torch.nn as nn
import  torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, num_classes, linear=True):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    if linear:
      self.out = nn.Linear(hidden_size, num_classes)
    else:
      self.out = NALU(128, 100, num_classes, 2)
    
  def forward(self, input): 
    # Set initial hidden and cell states 
    hidden = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device) 
    cell_state = torch.zeros(self.num_layers, input.size(0), self.hidden_size).to(device)
    # Forward propagate LSTM
    output, _ = self.lstm(input, (hidden, cell_state))  # out: tensor of shape (batch_size, seq_length, hidden_size)
    # Decode the hidden state of the last time step
    output = self.out(output[:, -1, :])
    return output
