# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from json import load, dump

# Local python scripts
from helpers import roll_avg_rel_change, calc_label, numel
from models import CNN, FCNN, Quadratic

# Global constants
learning_rate = 0.01
batch_size = 64
max_epochs = 500
window = 10 # Window size for convergence crit.
rel_conv_crit = 0.01
abs_conv_crit = 0.01

# Input shape
channels = 3 # RGB images
img_size = 8 # Image size length
input_shape = (channels, img_size, img_size)
input_size = img_size * img_size * 3

# Data size
N_tr = 10000
N_te = 10000

# Full training and test setas
gauss_x_tr = torch.tensor(np.random.normal(0,1,size=(N_tr,*input_shape)),dtype=torch.float32)
gauss_x_te = torch.tensor(np.random.normal(0,1,size=(N_te,*input_shape)),dtype=torch.float32)

# Full h1 training and test labels (Replace with p=1 for h1)
gauss_y_tr = calc_label(gauss_x_tr, p=2)
gauss_y_te = calc_label(gauss_x_te, p=2)

# Train models on various input sizes
for N in np.vectorize(int)(np.array(np.logspace(2, 4, 10))):
    
    print(f"Training set size: {N}")
    
    # Models
    CNNreLU, CNNquad = CNN(input_shape, nn.ReLU()), CNN(input_shape, Quadratic())
    FCNNreLU, FCNNquad = FCNN(input_size, nn.ReLU()), FCNN(input_size, Quadratic())
    models = [CNNreLU, CNNquad, FCNNreLU, FCNNquad]
    names = ["2-CNN+ReLU", "2-CNN+Quad","2-FCNN+ReLU","2-FCNN+Quad"]
    
    # Optimizers
    optimizers = [optim.SGD(model.parameters(), lr=learning_rate) for model in models]
    criterion = nn.BCELoss()
    
    # Create a DataLoader for dataset
    dataset = TensorDataset(gauss_x_tr[:N], gauss_y_tr[:N])
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for i, model in enumerate(models):
        optimizer = optimizers[i]
        model.train()
        epoch = 0
        converged = False
        queue = []
        while not converged:
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                # Is it converged?
                roll_avg = roll_avg_rel_change(queue, window, loss.item())
                if (roll_avg and roll_avg < rel_conv_crit and loss < abs_conv_crit) or epoch == max_epochs:
                    converged = True
                    break

            epoch+=1
        
        print(names[i], "finished. \tEpoch:", epoch,"\tLoss: %.2f" % loss.item(), "\tRoll: %.2f" % roll_avg)

    # Evaluate models
    test_loss = [0 for i in range(len(models))]
    accuracy = [0 for i in range(len(models))]
    
    with torch.no_grad():
        for i, model in enumerate(models):
            model.eval()
            out = model(gauss_x_te)
            test_loss[i] += criterion(out, gauss_y_te)
            accuracy[i] += float(sum(torch.eq((out>0.5).to(float),gauss_y_te))/N_te)

    # Read the JSON file
    file_path = 'test_acc.json'
    with open(file_path, 'r') as json_file:
        test_acc = load(json_file)

    # Add experiment to results
    test_acc[str(N)]=accuracy

    # Write accuracy to file
    with open(file_path, 'w') as json_file:
        dump(test_acc, json_file)
