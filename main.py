import time
import os

import allium 
import torch

import pickle
from datetime import *

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

def main(priormin, priormax):
    """
    Basic call script for running cell migration inference using active brownian particle model
    """
    # Preparing simulations
    #general options
    log = False
    test = True
    save_prob = 0
    nruns = 1000
    nproc = 16
    #io data
    posterior_file = 'posterior'
    posterior_folder = 'posteriors/5params/'
    if not os.path.exists(posterior_folder): os.makedirs(posterior_folder) 
    outputfolder = 'output/5params/'
    if not os.path.exists(outputfolder): os.makedirs(outputfolder);os.makedirs(f'{outputfolder}/data');

    # simulation data (inc parameter and ranges)
    d = {'factive':'v0', 'pairstiff':'k', 'tau':'tau', 'divrate': 'a'}
    priormin = [30,20,1, 4e-4]
    priormax = [150,150,10, 8e-3]
    prior = allium.utils.init_prior([priormin, priormax])
    starttime = 0
    endtime = 320    
    # summary statistics to calculate
    ssopts = ['A','B','C','D','E']
    #posterior options
    posterior_opt = ['flow','mix']

    #Prepare simulation object
    sim = allium.simulate.Simulate(pmap = d,\
                    ssopts=ssopts,\
                    log=log, \
                    test=test, \
                    folder=outputfolder, \
                    save_prob=0, \
                    starttime = 0, \
                    endtime = 320)
    simulator, prior = prepare_for_sbi(sim.wrapper, prior)

    # Running simulations
    print('Beginning simulation rounds')
    tic = time.perf_counter() 

    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=nruns, num_workers = nprocs)
    # Save parameter/observable data
    try:
        picklefile = open(f'{outputfolder}/data/run_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.p', 'wb') 
    pickle.dump(mix_posterior, picklefile)
    toc = time.perf_counter()
    print(f"Completed simulations in {toc - tic:0.4f} seconds")

    # Run inference
    if 'flow' in posterior_opt:
        flow_density_estimator_build_fun = posterior_nn(model='maf', hidden_features=60, num_transforms=3)
        flow_inference = SNPE( prior, density_estimator=flow_density_estimator_build_fun)
        flow_density_estimator = flow_inference.append_simulations(theta, x).train()
        flow_posterior= flow_inference.build_posterior(flow_density_estimator) 
        picklefile = open(f'{posterior_folder}/flow{posterior_file}.p', 'wb') 
        pickle.dump(flow_posterior, picklefile)

    if 'mix' in posterior_opt:
        mix_inference = SNPE( prior, density_estimator='mdn')
        mix_density_estimator = mix_inference.append_simulations(theta, x).train()
        mix_posterior = mix_inference.build_posterior(mix_density_estimator)  
        picklefile = open(f'{posterior_folder}/mix{posterior_file}.p', 'wb') 
        pickle.dump(mix_posterior, picklefile)
    
    toc = time.perf_counter()
    print(f"Completed inference in {toc - tic:0.4f} seconds")

    return 0

if __name__ == "__main__":
    #v0, k, tau = [30,150], [20,150], [1,10], [1,10], [0.0,1] <- parameter bounds
    # additional:
    #a = [4e-4, 8e-3] (reflects a/d0 of [0.01 to 0.2])

    priormin = [30,20,1, 4e-4]
    priormax = [150,150,10,8e-3]

    main()
