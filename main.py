import time
import os
import argparse
import json

import allium 
import torch

import pickle

from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn

def main(args):
    """
    Basic call script for running cell migration inference using active brownian particle model
    """
    # Preparing simulations
    if args.simulate:
        if args.save_prob:
            print('saving data')
        sim = allium.simulate.Sim()
        obs = sim.simulate(args.theta)
        ssvect, ssdata = allium.summstats.calculate_summary_statistics(obs,opts = self.ssopts,log = self.log, starttime=self.starttime, endtime=self.endtime)
        if args.save_prob == 1:
            picklefile = open(f'{args.outputfile}.p', 'wb') 
            pickle.dump(obs, picklefile)                    
            with open(f'{args.outputfile}_ss.p','wb') as f:
                pickle.dump(ssdata, f)
        return sim,ssvect,ssdata

    if not os.path.exists(args.posteriorfolder): os.makedirs(args.posteriorfolder) 
    if not os.path.exists(args.outputfolder): 
        os.makedirs(args.outputfolder)
        os.makedirs(f'{outputfolder}/data');
    d =json.loads(args.thetadict)
    prior = allium.utils.init_prior([args.thetamin[:len(d)], args.thetamax[:len(d)]])
    posterior_opt = ['flow','mix']
    print('Preparing simulator')
    #Prepare simulation object
    sim = allium.simulate.Sim(pmap = d,\
                    run = args.outputfile,\
                    ssopts=args.summstatsopts,\
                    log=args.log, \
                    test=args.test, \
                    folder=args.outputfolder, \
                    save_prob=args.save_prob, \
                    starttime = args.starttime, \
                    endtime = args.endtime)
    simulator, prior = prepare_for_sbi(sim.wrapper, prior)

    # Running simulations
    print('Beginning simulation rounds')
    tic = time.perf_counter() 

    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=args.nruns, num_workers = args.nprocs)
    # Save parameter/observable data
    picklefile = open(f'{args.outputfolder}/data/{args.outputfile}_{time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())}.p', 'wb') 
    pickle.dump((theta,x), picklefile)
    toc = time.perf_counter()
    print(f"Completed simulations in {toc - tic:0.4f} seconds")

    # Run inference
    if 'flow' in args.posterioropt:
        flow_density_estimator_build_fun = posterior_nn(model='maf', hidden_features=60, num_transforms=3)
        flow_inference = SNPE( prior, density_estimator=flow_density_estimator_build_fun)
        flow_density_estimator = flow_inference.append_simulations(theta, x).train()
        flow_posterior= flow_inference.build_posterior(flow_density_estimator) 
        picklefile = open(f'{args.posteriorfolder}/flow{args.pfile}.p', 'wb') 
        pickle.dump(flow_posterior, picklefile)

    if 'mix' in args.posterioropt:
        mix_inference = SNPE( prior, density_estimator='mdn')
        mix_density_estimator = mix_inference.append_simulations(theta, x).train()
        mix_posterior = mix_inference.build_posterior(mix_density_estimator)  
        picklefile = open(f'{args.posteriorfolder}/mix{args.pfile}.p', 'wb') 
        pickle.dump(mix_posterior, picklefile)
    
    toc = time.perf_counter()
    print(f"Completed inference in {toc - tic:0.4f} seconds")
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Running cell migration inference using active brownian particle models and neural network approximators')
    #general options
    parser.add_argument('--sim', dest='simulate', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--no-test', dest='test', action='store_false')
    parser.add_argument('--log', dest='log', action='store_true',help='Testing summary statistics using previously saved file')
    parser.add_argument('--no-log', dest='log', action='store_false')
    parser.add_argument('-s','--save_prob',type = float, default = 0, help='(float)\nProbability of saving file')
    parser.add_argument('-nruns','--nruns',type = int, default = 1000, help='(int)\nNumber of simulations to run')
    parser.add_argument('-nprocs','--nprocs',type = int, default = 16, help='(int)\nNumber of cores to use')
    #io data
    parser.add_argument('-ofo','--outputfolder', default = 'output/', help='(str)\nFolder for data')
    parser.add_argument('-ofi','--outputfile', default = 'run0', help='(str)\nFolder for data')
    parser.add_argument('-pfo','--posteriorfolder', default ='posteriors/', help='(str)\nFolder for posteriors')
    parser.add_argument('-pfi','--pfile', default = 'posterior', help='(str)\nFilename for posteriors')
    # simulation data (inc parameter and ranges) 
    parser.add_argument('-theta', '--theta', nargs = '+', default = [],help='(list)\nList of parameter values to pass to simulation')
    parser.add_argument('-c', '--configfile', default = "include/config/simconfig.json")
    parser.add_argument('-d', '--thetadict', type=str, default = '{"factive":"v0", "pairstiff":"k", "tau":"tau"}', help='(dict)\nDictionary mapping simulation parameters to passed parameters')
    parser.add_argument('-thetamin','--thetamin', nargs = '+', default = [30,20,1, 4e-4,0],help='(list)\nList of lowerbound parameters values')
    parser.add_argument('-thetamax','--thetamax', nargs = '+', default = [150,150,10, 8e-3,1], help='(list)\nList of upperbound parameters values')
    parser.add_argument('-start','--starttime',type = int, default = 60, help='(int)\nStarting frame number for summary statistics')
    parser.add_argument('-end','--endtime',type = int, default = 320, help='(int)\nFinal frame number for summary statistics')
    # summary statistics to calculate
    parser.add_argument('-ssopts','--summstatsopts', nargs = '+', default = ['A','B','C','D', 'E', 'F','G'], help='(list)\nSummary statistics to calculate (see allium/summstats.py for more details')
    #posterior options
    parser.add_argument('--pcalc', dest='pcalc', help = '(bool)\nCalculating posterior based on simulation run')
    parser.add_argument('--non-pcalc', dest='pcalc')
    parser.add_argument('-popt', '--posterioropt', default = ['flow', 'mix'], help = '(list)\nList of posterior architectures to use')
    parser.set_defaults(log=False,sim = False,test=False,pcalc=True)

    args = parser.parse_args()

    main(args)
