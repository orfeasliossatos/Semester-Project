# Implementation of the experiments described in Week2-TrainsetVSTestloss

# Optional Arguments
# -actv (str) The name of the activation function to use
# -arch (str) The name of the architecture to use
# -min/max (int) The min/max image side length.
# -f (str) The output .json file name (must contain an empty {}). Default week2_out.pkl
# -e (int) The maximum number of epochs to run the programme for. Default 100.
# -p (1/2) The p-norm used in the labelling function. Default 2.
# -acc (float) Goal accuracy for learning
# -del (bool) Whether or not to delete the results for the given (actv, arch) pair.
# -prop (float) If using ParamFairCNN then defines the proportionality factor.

# Imports
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Local python scripts
from helpers import roll_avg_rel_change, calc_label, printProgressBar, strToBool, train_model, count_parameters
from models import ModelLoader

# Read terminal input
args = sys.argv[1:]
arg_pairs=zip(args[::2],args[1::2])
valid_options = dict([('-actv', str), ('-arch', str), ('-min', int), ('-max', int), ('-f', str), ('-e', int), ('-p', int), ('-acc', float), ('-del', strToBool),  ('-prop', float)])

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

# Print options
print("Running with options", option_dict)

# Seed random number generation
torch.manual_seed(0)
np.random.seed(0)

# Model loader
loader = ModelLoader()

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, "\n")

# Don't try to use CuDNN with Tesla GPU
torch.backends.cudnn.enabled = False

# Empty cache because GPU has not a lot of memory
torch.cuda.empty_cache()

# Set this environment variable to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


# Global constants
learning_rate = 0.01
batch_size = 64
max_epochs = option_dict.get('-e') or 100
window = 10 # Window size for convergence crit.
rel_conv_crit = 0.01
abs_conv_crit = 0.01
epsilon = option_dict.get('-acc')/100.0 or 0.7 # Required accuracy
tolerance = 0.01 # Required tolerance
proportion = option_dict.get('-prop') or None 

# Input shape
channels = 3 # RGB images
img_sizes =  np.arange(option_dict.get('-min') or 4, option_dict.get('-max') or 14) # Image side lengths
input_sizes = 3*img_sizes**2 # Input dimension
input_shapes = [(channels, img_size, img_size) for img_size in img_sizes]

# Full dataset sizes
N_tr = 10000
N_te = 10000

# Get architecture name
architecture = option_dict.get('-arch') or "FCNN"
activation = option_dict.get('-actv') or "ReLU"
arch_name = architecture + "+" + activation + (str(proportion) if proportion else "" )

# Create file if it doesn't exist
file_path = option_dict.get('-f') or 'week2_out.pkl'
with open(file_path, 'ab+') as file:
    if os.stat(file_path).st_size == 0:
        pickle.dump(dict(), file)

# Option del - delete contents of results file
if option_dict.get('-del') or False:
    with open(file_path, 'rb+') as file:
        results = pickle.load(file)

    # Filter out results with same name
    with open(file_path, 'wb+') as file:
        filtered = {(name, size) : val for (name, size), val in results.items() if name!=arch_name}
        pickle.dump(filtered, file)

# Print empty progress bar
print(f"Progress: 0 / {len(input_sizes)}")

# For increasing input dimension
for i, input_size in enumerate(input_sizes):
    
    # Full training and test sets
    gauss_x_tr = torch.tensor(np.random.normal(0,1,size=(N_tr,*(input_shapes[i]))),dtype=torch.float32)
    gauss_x_te = torch.tensor(np.random.normal(0,1,size=(N_te,*(input_shapes[i]))),dtype=torch.float32)

    # Full h2 training and test labels
    p_norm = option_dict.get('-p') or 2
    gauss_y_tr = calc_label(gauss_x_tr, p=p_norm)
    gauss_y_te = calc_label(gauss_x_te, p=p_norm)
    
    # print('After pushing data to GPU: ', torch.cuda.memory_stats(device)['allocated_bytes.all.current'])
  
    
    # Models
    model_options = {'input_shape': input_shapes[i], 'proportion': proportion}
    model = loader.load(architecture, activation, model_options).to(device)
    
    # Print number of parameters
    print("The model has", count_parameters(model), "parameters")
    print("Memory usage after creating model",torch.cuda.memory_stats(device)['allocated_bytes.all.current'])
    
    # Save initial model state
    torch.save(model.state_dict(), 'weights/'+arch_name+'.pth')
    
    # Find exact training sample set size by bisection
    found   = False
    n_curr  = np.random.randint(100, 1000)
    n_hist  = [0] 
    n_prev  = 0
    iterate = 1 
    while not found:
        print("Iterate ", iterate, "Training samples: ", n_curr)
        
        # Reset model
        model.load_state_dict(torch.load('weights/'+arch_name+'.pth'))
        
        # Train model
        _, accuracy = train_model(model, batch_size, learning_rate, gauss_x_tr[:n_curr], gauss_y_tr[:n_curr], gauss_x_te, gauss_y_te, rel_conv_crit, abs_conv_crit, max_epochs, window, N_te, device)
        found = abs(accuracy - epsilon) < tolerance
                
        print("Finished training. Acc: ", accuracy, "n =", n_curr)
        
        # Finish if found.
        if found:
            print(f"{arch_name} converged after {iterate} iterations (dimension={input_size}).")
            break

        # Bisection method for finding correct training set size
        n_hist += [n_curr]
        n_hist.sort()
        idx = n_hist.index(n_curr)
        if accuracy > epsilon:
            n_curr = n_curr // 2 if idx == 0 else (n_curr + n_hist[idx-1]) // 2
        else:
            n_curr = 2 * n_curr if idx == len(n_hist)-1 else (n_curr + n_hist[idx+1]) // 2

        # If converged (not found) then reset the history and n_curr
        if abs(n_prev - n_curr) < 4:
            n_curr  = np.random.randint(100, 1000)
            n_hist  = [0] 
            n_prev  = 0
        else:
            n_prev = n_curr

        # Try again with a different number of training samples
        iterate += 1
    
    # Read the JSON file
    with open(file_path, 'rb') as file:
        results = pickle.load(file)

    # Add experiment to results
    results[(arch_name, input_size)] = n_curr

    # Write training set size to file
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)
        
    # Print progress
    print(f"Progress: {i+1} / {len(input_sizes)}")
