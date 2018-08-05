class NAC(nn.Module):
  '''
    Neural Architecture Cell
  '''
  def __init__(self, input_size, output_size):
    '''
      input_size: input dimension
      output_size: output dimension
    '''
    super(NeuralArchitectureCell, self).__init__()
    self.input = input_size
    self.output = output_size
    
    self.W_h = Parameter(torch.Tensor(self.output, self.input, device=device))
    self.M_h = Parameter(torch.Tensor(self.output, self.input, device=device))
    self.W = Parameter(F.tanh(self.W_h) * F.sigmoid(self.M_h))
    self.register_parameter('bias', None)
    
    init.xavier_uniform_(self.W_h)
    init.xavier_uniform_(self.M_h)
  
  def forward(self, input):
    return F.linear(input, self.W, self.bias)
