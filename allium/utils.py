import numpy as np
import pandas as pd
import glob
import json
import torch
from sbi import utils as sbiutils

import contextlib
import joblib

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallBack(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallBack
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

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
