import torch.nn as nn
import torch.optim as optim

# Define CNN
class CNN(nn.Module):
    def __init__(self, activation):
        super(CNN, self).__init__()
        
        self.conv = nn.Conv2d(3, 10, kernel_size=3, stride=1, padding=1)
        self.activation = activation
        self.fc = nn.Linear(160, 1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid = nn.Sigmoid()
        
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        self.fc = nn.Linear(x.size(1), 1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
# Define FCNN
class FCNN(nn.Module):
    def __init__(self, input_size, activation):
        super(FCNN, self).__init__()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(input_size,3072)
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
