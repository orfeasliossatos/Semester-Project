import torch
import numpy as np

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
        return np.mean((nqueue[1:] - nqueue[:-1]) / nqueue[:-1])

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