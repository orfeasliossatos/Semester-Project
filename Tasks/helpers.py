import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

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

def calc_label(x, p):
    """
        Computes canonical label over input tensor x that indicates
        whether the p-norm of the Red channel is larger than the Green one.
        
        Parameters:
            x: tensor of shape (N, 3, k, k) 
            p: the norm
        Returns:
            y: the label tensor of shape (N, 1)
    """
    cumul_x = torch.sum(x**p, dim=(2,3))
    y = (cumul_x.T[0] > cumul_x.T[1]).to(torch.float32)
    return y.view(y.size(0),1)

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

def train_model(model, batch_size, learning_rate, x_tr, y_tr, x_te, y_te, rel_conv_crit, abs_conv_crit, max_epochs, window, N_te, do_print=True):
    """
    Performs a training routine on a model with SGD, BCELoss, until 
    the relative or absolute convergence criteria are met or until
    the maximum number of epochs is reached. 
    @params:
        (TODO)
    """
    # Create dataloaders
    dataset = TensorDataset(x_tr, y_tr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Optimizer and criterion
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Training loop
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

        epoch += 1
        if epoch % 10 == 0 % do_print:
            print("Epoch:", epoch, "\tRolling Average Loss:", roll_avg, "\tLoss:", loss.item())
    
    # Evaluate models
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        model.eval()
        out = model(x_te)
        test_loss += criterion(out, y_te)
        accuracy += float(sum(torch.eq((out>0.5).to(float),y_te))/N_te)
    
    return test_loss, accuracy
