import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from itertools import product
from torch.utils.data import TensorDataset, DataLoader

def read_options(valid, args):
    """
        Takes a map of valid types {'-name' : type} and the program arguments,
        and tries to convert the program arguments to the corresponding
        option types, returning a dictionary of parameters {'-name' : value}.

        If the conversion fails, returns a ValueError.
    """
    # Zip up the arguments into subsequent pairs of two
    arg_pairs=zip(args[::2],args[1::2])
    
    # The dictionary of parameters to return
    option_dict = {}
    option_dict.setdefault(None)

    # Try convert the parameters.
    try:
        if len(args) % 2 != 0:
            print("Invalid arguments. Must be an even number of values.")
            exit()

        for i, (arg, val) in enumerate(arg_pairs):
        
            if arg not in valid.keys():
                raise ValueError("Option unavailable:", arg) 
        
            else:
                # Try convert to type
                option_dict[arg]=valid[arg](val)
         
    except ValueError:
        print(ValueError)

    return option_dict

def roll_avg_rel_change(queue, window, new):
    """
        Computes the rolling average of relative changes
        between subsequent elements of a queue if it is
        of length at least window, otherwise returns None.
        
        Parmeters:
            queue: a FIFO queue of floats
            window: the maximum queue length
            old_avg: the previous computed average
            new: the newest float to be added to the queue
        Returns:
            The new rolling average.
    """
    queue.insert(0, new)
    
    # If not a fully formed queue
    if len(queue) <= window:
        return None
    else:
        queue.pop()
        nqueue = np.array(queue)
        
        # Return average of relative changes
        return np.mean(abs((nqueue[1:] - nqueue[:-1]) / nqueue[:-1]))


def numel(shape, transforms):
    """
        Returns the number of elements a tensor of the given
        shape after a list of transforms applied in order of 
        increasing index value.
        
        @params:
            shape: a tuple (d1,d2,d3,...)
            transforms: a list of tensor transforms 
        Returns:
            The number of elements in the transformed tensor.
    """
    
    y = torch.tensor(np.random.normal(0,1,size=shape),dtype=torch.float32)
    for t in transforms:
        y=t(y)
        
    return torch.numel(y)

def count_parameters(model): 
    return sum(p.numel() for p in model.parameters() if p.requires_grad) 

def standardize(arr):
    """
        Returns a standardized array.
    """
    return (arr - arr.mean()) / np.maximum(arr.std(), 1e-8)

def min_max_rescale(arr, a, b):
    """
        Returns a min-max rescaled array so that its values are between a and b.
    """
    if (len(set(arr.flatten()))==1):
        return arr
    else:
        return a + ((b - a) * (arr - np.min(arr))) / (np.max(arr) - np.min(arr))

def permute_along_axes(arr, axes):
    """
        Takes a numpy array and permutes its values along the specified axes.
    """
    rng = np.random.default_rng()
    for ax in axes:
        rng.permuted(arr, axis=ax, out=arr)


def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = '*' * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}\r', end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()

def strToBool(string):
    """
    Parse a string as a boolean.
    @params:
        string     - Required : the string to parse (Str)
    """
    return True if string=="True" else False if string=="False" else False

def train_model(model, batch_size, learning_rate, x_tr, y_tr, x_te, y_te, rel_conv_crit, abs_conv_crit, max_epochs, window, N_te, device, do_print=True):
    """
    Performs a training routine on a model with SGD, BCELoss, until 
    the relative or absolute convergence criteria are met or until
    the maximum number of epochs is reached. 
    @params:
        (TODO)
    """
    # Create dataloaders
    tr_dataset = TensorDataset(x_tr, y_tr)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
    
    te_dataset = TensorDataset(x_te, y_te)
    te_loader = DataLoader(te_dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    model.train()
    epoch = 0
    converged = False
    queue = []
    while not converged:
        for batch_x, batch_y in tr_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            tr_loss = criterion(output, batch_y)
            tr_loss.backward()
            optimizer.step()

            # Is it converged?
            roll_avg = roll_avg_rel_change(queue, window, tr_loss.item())
            if (roll_avg and roll_avg < rel_conv_crit and tr_loss < abs_conv_crit) or epoch == max_epochs:
                converged = True
                break

        epoch += 1
        if epoch % 50 == 0 % do_print:
            print("Epoch:", epoch, "\tRolling Average Loss:", roll_avg, "\tLoss:", tr_loss.item())
    
    # Evaluate models
    te_loss = 0
    accuracy = 0
    with torch.no_grad():
        model.eval()
        
        # Batch to lower GPU memory usage
        for batch_x, batch_y in te_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            output = model(batch_x)
            te_loss += criterion(output, batch_y)
            accuracy += sum(torch.eq((output>0.5).to(float), batch_y))
    
    return te_loss, float(accuracy / N_te)


def unpack_and_aggregate(results, model_name, ops):
    
    # Unzipped lists
    xs, ys = list(zip(*sorted([(int(dim), tr_size) for (_name, dim), tr_size  in results.items() if model_name == _name])))

    # Format lists for scattering
    xs_rep = []
    for i, y in enumerate(ys):
        xs_rep = np.concatenate([xs_rep, np.repeat(xs[i], len(y))])

    ys_flat = np.array(ys).flatten()

    # Operations
    results = [[op(y) for y in ys] for op in ops]

    return xs_rep, ys_flat, *tuple(results)