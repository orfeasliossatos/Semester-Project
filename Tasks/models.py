import torch.nn as nn
import torch.optim as optim
from numpy import prod
from helpers import numel

# Define CNN
class CNN(nn.Module):
    def __init__(self, activation, model_options):
        super(CNN, self).__init__()
        
        self.conv = nn.Conv2d(3, 100, kernel_size=3, stride=1, padding=1)
        self.activation = activation
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(numel(model_options.get('input_shape'), [self.conv,self.pool]), 1)
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
class DumbCNN(nn.Module):
    def __init__(self, activation, model_options):
        super(DumbCNN, self).__init__()
        
        self.conv = nn.Conv2d(3, 2, kernel_size=1, stride=1, padding=0)
        self.activation = activation
        self.pool = nn.AvgPool2d(model_options.get('input_shape')[1])
        self.fc = nn.Linear(numel(model_options.get('input_shape'), [self.conv,self.pool]), 1)
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
    def __init__(self, activation, model_options):
        super(FCNN, self).__init__()
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(prod(model_options.get('input_shape')),3072)
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
    
# Define a DumbCNN that's comparable to a FCNN in parameters up to some scalar factor.
class ParamFairCNN(nn.Module):

    # Note - number of parameters with in_channels=3, kernel_size=1, stride=1, padding=0:
    # 5 * out_channels + 1 = req_parameters
    # Reason for 5 : 3 for in_channels + 1 bias for each out_channel + 1 for linear layer
    def __init__(self, activation, model_options):
        super(ParamFairCNN, self).__init__()
        prop = model_options.get('proportion') or 1.0
        req_params = prop * sum(p.numel() for p in FCNN(activation, model_options).parameters())
        out_channels = int((req_params - 1) / 5)
        self.conv = nn.Conv2d(3, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = activation
        self.pool = nn.AvgPool2d(model_options.get('input_shape')[1])
        self.fc = nn.Linear(numel(model_options.get('input_shape'), [self.conv,self.pool]), 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Quadratic activation function
class Quadratic(nn.Module):
    def forward(self, x):
        return x**2

# Loads models from strings
class ModelLoader:
    def __init__(self):
        self.architectures = dict(zip(["CNN", "DumbCNN", "FCNN", "ParamFairCNN"], [CNN, DumbCNN, FCNN, ParamFairCNN]))
        self.activations = dict(zip(["ReLU", "Quad"],[nn.ReLU(), Quadratic()]))

    def load(self, architecture, activation, model_options):
        """
        Loads an architecture with options.
        @params:
            architecture    - Required : name of the architecture (Str)
            activation      - Required : name of the activation function (Str)
            input_shape     - Optional : shape of a single input (Tuple)
        """
        return self.architectures[architecture](self.activations[activation], model_options)
