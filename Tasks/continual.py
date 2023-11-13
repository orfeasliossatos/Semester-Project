# Imports
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Local python scripts
from helpers import read_options, roll_avg_rel_change, calc_label, printProgressBar, strToBool, train_model, count_parameters
from models import ModelLoader

# Read terminal input
args = sys.argv[1:]

valid = dict([('-min', int), ('-max', int), ('-f', str), ('-e', int), ('-acc', float), ('-del', strToBool)])

option_dict = read_options(valid, args)

