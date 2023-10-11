# Implementation of the experiments described in Week1-FCNNvsCNN

# Optional Arguments
# -actv (str) The name of the activation function to use
# -min/max (int) The min/max order of magnitude for trainset size. Default 2/4.
# -s (int) The number training sample sets (of exponentially increasing size). Default 5.
# -f (str) The output .json file name (must contain an empty {}). Default week1_out.json
# -e (int) The maximum number of epochs to run the programme for. Default 100.
# -p (1/2) The p-norm used in the labelling function. Default 2.
# -l (int) Image side-length. Default 8.

# Imports
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

# Local python scripts
from helpers import roll_avg_rel_change, calc_label, numel, printProgressBar, strToBool, train_model
from models import ModelLoader

# Read terminal input
args = sys.argv[1:]
arg_pairs=zip(args[::2],args[1::2])
valid_options = dict([('-actv', str), ('-arch', str), ('-min', int), ('-max', int), ('-s', int), ('-f', str), ('-e', int), ('-p', int), ('-l', int), ('-del', strToBool), ('-prop', float)])

# Save options here
option_dict = {}
option_dict.setdefault(None)

try:
    if len(args) % 2 != 0:
        print("Invalid arguments. Must be an even number of values.")
        exit()

    for i, (arg, val) in enumerate(arg_pairs):
    
        if arg not in valid_options.keys():
            raise ValueError("Option unavailable:", arg) 
    
        else:
            # Try convert to type
            option_dict[arg]=valid_options[arg](val)
         
except ValueError:
    print(ValueError)
    
# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, "\n")

# Seed random number generation
torch.manual_seed(0)
np.random.seed(0)

# Model loader
loader = ModelLoader()

# Global constants
learning_rate = 0.01
batch_size = 64
max_epochs = option_dict.get('-e') or 20
window = 10 # Window size for convergence crit.
rel_conv_crit = 0.01
abs_conv_crit = 0.01

# Input shape
channels = 3 # RGB images
img_size = option_dict.get('-l') or 8 # Image size length
input_shape = (channels, img_size, img_size)
input_size = img_size * img_size * 3

# Data size
min_mag = option_dict.get('-min') or 2
max_mag = option_dict.get('-max') or 4
splits = option_dict.get('-s') or 5
N_tr = 10**max_mag
N_te = 10000
Ns = np.vectorize(int)(np.array(np.logspace(min_mag, max_mag, splits)))

# Full training and test setas
gauss_x_tr = torch.tensor(np.random.normal(0,1,size=(N_tr,*input_shape)),dtype=torch.float32, device=device)
gauss_x_te = torch.tensor(np.random.normal(0,1,size=(N_te,*input_shape)),dtype=torch.float32, device=device)

# Full h1 training and test labels (Replace with p=1 for h1)
p_norm = option_dict.get('-p') or 2
gauss_y_tr = calc_label(gauss_x_tr, p=p_norm).to(device)
gauss_y_te = calc_label(gauss_x_te, p=p_norm).to(device)

# Model name
architecture = option_dict.get('-arch') or "FCNN"
activation = option_dict.get('-actv') or "ReLU"
arch_name = architecture + "+" + activation

# Create file if it does not exist
file_path = option_dict.get('-f') or 'week1_out.pkl'
with open(file_path, 'ab+') as file:
    if os.stat(file_path).st_size == 0:
        pickle.dump(dict(), file)

# Option del - delete contents of results file
if option_dict.get('-del') or False:
    with open(file_path, 'rb+') as file:
        test_acc = pickle.load(file)
    
    with open(file_path, 'wb+') as file:
        filtered = {(name, size) : val for (name, size), val in test_acc.items() if name!=arch_name}
        pickle.dump(filtered, file)

# Print empty progress bar
printProgressBar(0, splits, prefix = 'Progress:', suffix = 'Complete', length = 50)

# Train models on various input sizes
for k, N in enumerate(Ns):
    
    # Create model
    model_options = {'input_shape': input_shape, 'proportion': option_dict.get('-prop')}
    model = loader.load(architecture, activation, model_options)
    
    _, accuracy = train_model(model, batch_size, learning_rate, gauss_x_tr[:N], gauss_y_tr[:N], gauss_x_te, gauss_y_te, rel_conv_crit, abs_conv_crit, max_epochs, window, N_te)

    # Read file contents
    with open(file_path, 'rb') as file:
        test_acc = pickle.load(file)

    # Add experiment to results
    test_acc[(arch_name, N)] = accuracy

    # Write accuracy to file
    with open(file_path, 'wb') as file:
        pickle.dump(test_acc, file)

    # Print progress
    printProgressBar(k + 1, splits, prefix = 'Progress:', suffix = 'Complete', length = 50)
