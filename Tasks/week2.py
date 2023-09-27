# Imports
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
from json import dump, load

# Local python scripts
from helpers import roll_avg_rel_change, calc_label, numel, printProgressBar
from models import CNN, DCNN, FCNN, Quadratic

# Read terminal input
args = sys.argv[1:]
arg_pairs=zip(args[::2],args[1::2])
valid_options = dict([('-min', int), ('-max', int), ('-f', str), ('-e', int), ('-p', int)])

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
epsilon = 0.9 # Required accuracy
tolerance = 0.01 # Required tolerance

# Input shape
channels = 3 # RGB images
img_sizes =  np.arange(option_dict.get('-min') or 4, option_dict.get('-max') or 14) # Image side lengths
input_sizes = 3*img_sizes**2 # Input dimension
input_shapes = [(channels, img_size, img_size) for img_size in img_sizes]

# Full dataset sizes
N_tr = 1000000
N_te = 10000

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
    models = []
    for arch in [CNN, DCNN, FCNN]:
        for activation in [nn.ReLU(), Quadratic()]:
            models += [arch(input_shapes[i], activation).to(device)]
            
    names = ["CNN+ReLU", "CNN+Quad","DCNN+ReLU", "DCNN+Quad", "FCNN+ReLU","FCNN+Quad"]
    
    # Found training sample set sizes
    ns = [0 for model in models]
    
    for j, model in enumerate(models):
        print(names[j])
        # summary(model, input_shapes[i])
        torch.save(model.state_dict(), 'Weights/'+names[j]+'.pth')
        
        # Find exact training sample set size by bisection
        n = 5000 # Initial number of training samples
        found = False
        n_tried = [0]
        iterate=1
        while not found:
            print("Iterate ", iterate, "Training samples: ", n)
            
            # Reset model
            model.load_state_dict(torch.load('weights/'+names[j]+'.pth'))
            
            # Create dataloaders
            dataset = TensorDataset(gauss_x_tr[:n], gauss_y_tr[:n])
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # Optimizer
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            
            # Training loop
            criterion = nn.BCELoss()
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
                
                # Save this training set size
                if found:
                    ns[j]=n
                    
            print("Finished training. Acc: ", accuracy, "n =", n)
            
            # Bisection method for finding correct training set size
            n_tried+=[n]
            n_tried.sort()
            idx = n_tried.index(n)
            if accuracy > epsilon:
                n = n // 2 if idx == 0 else (n + n_tried[idx-1]) // 2
            else:
                n = 2 * n if idx == len(n_tried)-1 else (n + n_tried[idx+1]) // 2
                
            # Try again with a different number of training samples
            iterate += 1
    
    # Read the JSON file
    file_path = option_dict.get('-f') or 'week2_out.json'
    with open(file_path, 'r') as json_file:
        trainset_sizes = load(json_file)

    # Add experiment to results
    trainset_sizes[str(input_size)]=ns

    # Write training set size to file
    with open(file_path, 'w') as json_file:
        dump(trainset_sizes, json_file)
        
    # Print progress
    printProgressBar(i + 1, len(input_sizes), prefix = 'Progress:', suffix = 'Complete', length = 50)
