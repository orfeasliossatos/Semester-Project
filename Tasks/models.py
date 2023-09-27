import torch.nn as nn
import torch.optim as optim
from numpy import prod
from helpers import numel

# Define CNN
class CNN(nn.Module):
    def __init__(self, input_shape, activation):
        super(CNN, self).__init__()
        
        self.conv = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        self.activation = activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(numel(input_shape, [self.conv,self.pool]), 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
# Define 'dumb' CNN
class DCNN(nn.Module):
    def __init__(self, input_shape, activation):
        super(DCNN, self).__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=3, stride=1, padding=1)
        self.activation = activation
        self.pool = nn.AvgPool2d(input_shape[1])
        self.fc = nn.Linear(numel(input_shape, [self.conv,self.pool]), 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
# Define FCNN
class FCNN(nn.Module):
    def __init__(self, input_shape, activation):
        super(FCNN, self).__init__()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(prod(input_shape),3072)
        self.activation = activation
        self.fc2 = nn.Linear(3072, 1)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# Quadratic activation function
class Quadratic(nn.Module):
    def forward(self, x):
        return x**2
