import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(1, 1)  # Single input neuron and single output neuron

    def forward(self, x):
        return self.fc(x)

model = SimpleModel().to(device)
input_data = torch.tensor([[2.0]], device=device)  # A single input value of 2.0
output = model(input_data)

print("Input:", input_data)
print("Output:", output)

