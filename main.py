import os

import allium 
import torch

import random
import pickle
import json
import time
import subprocess
import numpy as np

from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference.base import infer
import matplotlib.pyplot as plt


def calculate_summary_statistics(output, log=False):
    """
    Calculates summary statistics.

    In the future, this will include SAMoSA analysis 
    Currently, calculates D from MSD, average horizontal distance, change in density
    """
    Lx = output['params']['Lx']
    Ly = output['params']['Ly']

    #Eventually from SAMOSA - for now, look at following three:
    zap = 320
    time = np.linspace(26560, 39840, int((39840-26560)/83))
    msd = allium.ss.calculate_msd(output["data"],tracers=True, beg=zap, end = len(output['data']))
    D = np.polyfit(np.log(time[1:]), np.log(msd[1:]), 1)[0]
    xi = allium.ss.average_horizontal_displacement(output["data"],tracers=True)
    deltaphi = allium.ss.change_in_phi(output)
    if not log:
        print(f'\n<x_i> = {xi}')
        print(f'\nD = {D}')
        print(f'\n$\Delta\Phi$ = {deltaphi}\n\n\n\n')
            
    return [D, xi, deltaphi]

def init_prior(bounds ,num_dim = 3):
    """
    Returns prior (currently 3-dimensional parameter space sampled w uniform)
    """
    prior_min = bounds[0]
    prior_max = bounds[1]
    return utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),high=torch.as_tensor(prior_max))

def simulation_wrapper(params):
    """
    Returns summary statistics from active particle model of cells.

    Summarizes the output of the simulator and converts it to `torch.Tensor`.
    """
    filename = f'output/v0_{int(params[0])}_k_{int(params[1])}_tau_{int(params[2])}.p'    

    obs = allium.simulate.sim(params, log=True)
    save = random.uniform(0,1) < 0.0095
    if save:
        with open(filename,'wb') as f:
            pickle.dump(obs, f)

    summstats = torch.as_tensor(calculate_summary_statistics(obs,log=True))
    if save:
        with open(filename,'wb') as f:
            obs['ss'] = summstats
            pickle.dump(obs, f)

    return summstats

def main():
    print('beginning run')
    #v0, k, tau = [30,150], [20,150], [1,10]
    tic = time.perf_counter()

    prior = init_prior([[30,20,1],[150,150,10]])
    posterior = infer(simulation_wrapper, prior, method='SNPE', num_simulations=10, num_workers=4)
    toc = time.perf_counter()
    print(f"Completed in {toc - tic:0.4f} seconds")
    picklefile = open('testposterior.p', 'wb') 
    pickle.dump(posterior, picklefile)


    print(f"Completed in {toc - tic:0.4f} seconds")

    return 0

if __name__ == "__main__":
    main()
