import torch
import torch.nn as nn
from torch import optim
from torch.nn.parameter import Parameter
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from models import *

def generate(train_size, test_size, fn):
  '''
    Generate dataset
    train_size: number of training rows
    test_size: number of test rows
    fn: function applied
  '''
  X, Y = torch.Tensor(train_size + test_size, 2, device=device), torch.Tensor(train_size + test_size, 1, device=device)
  for i in range(train_size + test_size):
    x = torch.rand(2, device=device)*10
    y = torch.tensor([fn(*x)], device=device)
    X[i] = x
    Y[i] = y
  X_train, y_train = X[:train_size], Y[:train_size]
  X_test, y_test = X[train_size:], Y[train_size:]
  return X_train, y_train, X_test, y_test

def train(model, optimizer, data, target, num_iters):
  all_losses = []
  for i in range(num_iters):
    out = model(data)
    loss = F.mse_loss(out, target)
    mea = torch.mean(torch.abs(target - out))
    all_losses.append(loss)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 1000 == 0:
      print("\t{}/{}: loss: {:.7f} - mea: {:.7f}".format(i+1, num_iters, loss.item(), mea.item()))
  return all_losses

def test(model, data, target):
  with torch.no_grad():
    out = model(data)
    return out

def main():
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
  # define functiom
  fn = lambda x, y: x * y
  # generate datasets
  X_train, y_train, X_test, y_test = generate(500, 50, fn)
  
  model = NALU(2,2,1,2)
  optim = torch.optim.RMSprop(model.parameters(), lr = 1e-3)
  # train
  all_losses = train(model, optim, X_train, y_train, 100000)
  
  # plot losses
  plt.figure()
  plt.plot(all_losses)
  
  # test
  results = test(net, X_t, y_t)
  
  # check
  print(X_test[1], y_test[1], mse[1])

if __name__ == '__main__':
    main()
