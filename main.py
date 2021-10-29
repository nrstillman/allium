import time
import os
import argparse

import allium 
import torch

import pickle
from datetime import *

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

def main(args):
    """
    Basic call script for running cell migration inference using active brownian particle model
    """
    # Preparing simulations
    if not os.path.exists(posterior_folder): os.makedirs(posterior_folder) 
    if not os.path.exists(outputfolder): os.makedirs(outputfolder);os.makedirs(f'{outputfolder}/data');
    prior = allium.utils.init_prior([thetamin, thetamax])
    
    posterior_opt = ['flow','mix']

    #Prepare simulation object
    sim = allium.simulate.Simulate(pmap = d,\
                    ssopts=ssopts,\
                    log=log, \
                    test=test, \
                    folder=ofolder, \
                    save_prob=0, \
                    starttime = stime, \
                    endtime = etime)
    simulator, prior = prepare_for_sbi(sim.wrapper, prior)

    # Running simulations
    print('Beginning simulation rounds')
    tic = time.perf_counter() 

    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=nruns, num_workers = nprocs)
    # Save parameter/observable data
    picklefile = open(f'{outputfolder}/data/run_{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.p', 'wb') 
    pickle.dump((theta,x), picklefile)
    toc = time.perf_counter()
    print(f"Completed simulations in {toc - tic:0.4f} seconds")

    # Run inference
    if 'flow' in popt:
        flow_density_estimator_build_fun = posterior_nn(model='maf', hidden_features=60, num_transforms=3)
        flow_inference = SNPE( prior, density_estimator=flow_density_estimator_build_fun)
        flow_density_estimator = flow_inference.append_simulations(theta, x).train()
        flow_posterior= flow_inference.build_posterior(flow_density_estimator) 
        picklefile = open(f'{posterior_folder}/flow{posterior_file}.p', 'wb') 
        pickle.dump(flow_posterior, picklefile)

    if 'mix' in popt:
        mix_inference = SNPE( prior, density_estimator='mdn')
        mix_density_estimator = mix_inference.append_simulations(theta, x).train()
        mix_posterior = mix_inference.build_posterior(mix_density_estimator)  
        picklefile = open(f'{posterior_folder}/mix{posterior_file}.p', 'wb') 
        pickle.dump(mix_posterior, picklefile)
    
    toc = time.perf_counter()
    print(f"Completed inference in {toc - tic:0.4f} seconds")

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Running cell migration inference using active brownian particle models and neural network approximators')
    #general options
    parser.add_argument('log', default = False,help='(bool) \nRefers to whether output should be saved to log.txt')
    parser.add_argument('test', default = True,help='(bool)\nTesting summary statistics using previously saved file')
    parser.add_argument('save_prob', default = 0, help='(float)\nProbability of saving file')
    parser.add_argument('nruns', default = 1000, help='(int)\nNumber of simulations to run')
    parser.add_argument('nproc', default = 16, help='(int)\nNumber of cores to use')
    #io data
    parser.add_argument('ofolder', default = 'output/', help='(str)\nFolder for data')
    parser.add_argument('pfolder', default ='posteriors/', help='(str)\nFolder for posteriors')
    parser.add_argument('pfile', default = 'posterior', help='(str)\nFilename for posteriors')
    # simulation data (inc parameter and ranges) 
    parser.add_argument('d', default = {'factive':'v0', 'pairstiff':'k', 'tau':'tau', 'divrate': 'a'}, help='(dict)\nDictionary mapping simulation parameters to passed parameters')
    parser.add_argument('thetamin', default = [30,20,1, 4e-4],help='(list)\nList of lowerbound parameters values')
    parser.add_argument('thetamax', default = [150,150,10, 8e-3], help='(list)\nList of upperbound parameters values')
    parser.add_argument('stime', default = 0, help='(int)\nStarting frame number for summary statistics')
    parser.add_argument('etime', default = 320, help='(int)\nFinal frame number for summary statistics')
    # summary statistics to calculate
    parser.add_argument('ssopts', default = ['A','B','C','D','E'], help='(list)\nSummary statistics to calculate (see allium/summstats.py for more details')
    #posterior options
    parser.add_argument('pcalc', default = True, help = '(bool)\nCalculating posterior based on simulation run')
    parser.add_argument('popt', default = ['flow', 'mix'], help = '(list)\nList of posterior architectures to use')
    args = parser.parse_args()

    main(args)
