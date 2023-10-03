# Imports
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary

# Local python scripts
from helpers import roll_avg_rel_change, calc_label, printProgressBar, strToBool
from models import ModelLoader

# Read terminal input
args = sys.argv[1:]
arg_pairs=zip(args[::2],args[1::2])
valid_options = dict([('-actv', str), ('-arch', str), ('-min', int), ('-max', int), ('-f', str), ('-e', int), ('-p', int), ('-acc', float), ('-del', strToBool)])

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

# Seed random number generation
torch.manual_seed(0)
np.random.seed(0)

# Model loader
loader = ModelLoader()

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, "\n")

# Global constants
learning_rate = 0.01
batch_size = 64
max_epochs = option_dict.get('-e') or 100
window = 10 # Window size for convergence crit.
rel_conv_crit = 0.01
abs_conv_crit = 0.01
epsilon = option_dict.get('-acc') or 0.7 # Required accuracy
tolerance = 0.01 # Required tolerance

# Input shape
channels = 3 # RGB images
img_sizes =  np.arange(option_dict.get('-min') or 4, option_dict.get('-max') or 14) # Image side lengths
input_sizes = 3*img_sizes**2 # Input dimension
input_shapes = [(channels, img_size, img_size) for img_size in img_sizes]

# Full dataset sizes
N_tr = 1000000
N_te = 10000


# Option del - delete contents of results file
file_path = option_dict.get('-f') or 'week2_out.pkl'
if option_dict.get('-del') or False:
    with open(file_path, 'wb+') as file:
        pickle.dump(dict(), file)

# Print empty progress bar
printProgressBar(0, len(input_sizes), prefix = 'Progress:', suffix = 'Complete', length = 50)

# For increasing input dimension
for i, input_size in enumerate(input_sizes):
    
    # Full training and test sets
    gauss_x_tr = torch.tensor(np.random.normal(0,1,size=(N_tr,*(input_shapes[i]))),dtype=torch.float32, device=device)
    gauss_x_te = torch.tensor(np.random.normal(0,1,size=(N_te,*(input_shapes[i]))),dtype=torch.float32, device=device)

    # Full h2 training and test labels
    p_norm = option_dict.get('-p') or 2
    gauss_y_tr = calc_label(gauss_x_tr, p=p_norm).to(device)
    gauss_y_te = calc_label(gauss_x_te, p=p_norm).to(device)
    
    # Models
    architecture = option_dict.get('-arch') or "FCNN"
    activation = option_dict.get('-actv') or "ReLU"
    model = loader.load(architecture, activation, input_shape=input_shapes[i])
    name = architecture + "+" + activation
    summary(model, input_shapes[i])

    # Save initial model state
    torch.save(model.state_dict(), 'weights/'+name+'.pth')
    
    # Find exact training sample set size by bisection
    found   = False
    n_curr  = np.random.randint(100, 1000)
    n_hist  = [0] 
    n_prev  = 0
    iterate = 1 
    while not found:
        print("Iterate ", iterate, "Training samples: ", n_curr)
        
        # Reset model
        model.load_state_dict(torch.load('weights/'+name+'.pth'))
        
        # Create dataloaders
        dataset = TensorDataset(gauss_x_tr[:n_curr], gauss_y_tr[:n_curr])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Optimizer and criterion
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        # Training loop
        model.train()
        epoch = 0
        converged = False
        loss_queue = [] # For rolling training loss stop criterion
        while not converged:
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

                # Check for convergence
                roll_avg = roll_avg_rel_change(loss_queue, window, loss.item())
                if (roll_avg and roll_avg < rel_conv_crit and loss < abs_conv_crit) or epoch == max_epochs:
                    converged = True
                    break
            
            epoch += 1
        
        # Evaluate model
        with torch.no_grad():
            model.eval()
            out = model(gauss_x_te)
            test_loss = criterion(out, gauss_y_te)
            accuracy = float(sum(torch.eq((out > 0.5).to(float), gauss_y_te)) / N_te)
            found = abs(accuracy - epsilon) < tolerance
                
        print("Finished training. Acc: ", accuracy, "n =", n_curr)
        
        # Finish if found.
        if found:
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
        sample_complexity = pickle.load(file)

    # Add experiment to results
    sample_complexity[(name, input_size)] = n_curr

    # Write training set size to file
    with open(file_path, 'wb') as file:
        pickle.dump(sample_complexity, file)
        
    # Print progress
    printProgressBar(i + 1, len(input_sizes), prefix = 'Progress:', suffix = 'Complete', length = 50)