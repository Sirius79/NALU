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
import matplotlib.ticker as ticker
from models import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
def train(model, optimizer, train_loader, criterion, num_epochs):
  all_losses = []
  total_step = len(train_loader)
  for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
      images = images.reshape(-1, sequence_length, input_size).to(device)
      labels = labels.to(device)
        
      # Forward pass
      outputs = model(images)
      loss = criterion(outputs, labels)
      all_losses.append(loss)
      
      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
        
      if (i+1) % 100 == 0:
        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))
  return all_losses

# Test the model
def test(model, test_loader):
  # Test the model
  with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
      images = images.reshape(-1, sequence_length, input_size).to(device)
      labels = labels.to(device)
      outputs = model(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

def main():
  # hyper parameters
  sequence_length = 28
  input_size = 28
  hidden_size = 128
  num_layers = 2
  num_classes = 10
  batch_size = 32
  num_epochs = 5
  learning_rate = 0.001

  # get datasets
  train_dataset = torchvision.datasets.MNIST(root='../../data/', train=True, transform=transforms.ToTensor(), download=True)
  test_dataset = torchvision.datasets.MNIST(root='../../data/', train=False, transform=transforms.ToTensor())

  # Data loader
  train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
  
  # model
  model = RNN(input_size, hidden_size, num_layers, num_classes, False).to(device)
  
  # Loss and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
  
  # train
  all_losses = train(model, optimizer, train_loader, criterion, num_epochs)
  
  # plot losses
  plt.figure()
  plt.plot(all_losses)
  
  # test
  test(model, test_loader)
