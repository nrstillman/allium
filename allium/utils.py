import numpy as np
import pandas as pd
import glob
import json
import torch
from sbi import utils as sbiutils

def init_prior(bounds ): 
    """
    Returns prior 
    """
    prior_min = bounds[0]
    prior_max = bounds[1]
    return sbiutils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),high=torch.as_tensor(prior_max))

def read_output(f):
    """
    Reads in previously saved .dat files incase of debugging
    """
    build = True
    for line in open(f, 'r'):
        item = line.rstrip()
        if build:
            col = item.split(',')
            data = pd.DataFrame(columns = col)
            build = False
        else:
            row = pd.Series(item.split(','), index=col)
            data = data.append(row, ignore_index = True)
    return data

def read_params(configfile):
    """
    Reads in parameters from the config files
    """
    params = dict()
    with open(configfile) as jsonFile:
        parameters = json.load(jsonFile)
        for attribute in parameters:
            params[attribute] = parameters[attribute]
    return parameters
