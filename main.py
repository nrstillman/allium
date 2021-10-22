import os

import allium 
import torch

import random
import pickle
import json
import time
import subprocess
import numpy as np
from scipy import stats

from sbi import utils as utils
from sbi import analysis as analysis
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.inference.base import infer
from sbi.utils.get_nn_models import posterior_nn
import matplotlib.pyplot as plt


def calculate_summary_statistics(d, log=False,start=60,end=320,takeDrift=False, plot = False):
    """
    Calculates summary statistics.

    """
    # 0 is new cells, 1 is tracer, 2 is original (check this)
    usetypes = [0,1,2]
    end = int(d.param.zaptime/d.param.output_time) #320
    # remove any data post zap
    d.truncateto(starttime, endtime)
    ss = {}
    # # # # # A - Velocity distributions and mean velocity
    velbins=np.linspace(0,10,100)
    velbins2=np.linspace(-10,10,100)
    vav, vdist,vdist2 = allium.summstats.getVelDist(d, velbins,velbins2, usetype=usetypes,verbose=plot)
    ss['vav'] = vav
    ss['vdist'] = vdist
    ss['vdist2'] = vdist2
    # # B - Autocorrelation Velocity Function
    tval2, velauto, v2av = allium.summstats.getVelAuto(d, usetype=[1],verbose=plot)
    ss['tval2'] = tval2
    ss['velauto'] = velauto
    ss['v2av'] = v2av
    # C - Mean square displacement
    tval, msd, d = allium.summstats.getMSD(d,takeDrift, usetype=[1],verbose=plot)
    ss['tval'] = tval
    ss['msd'] = msd
    # # D - Self Intermediate Scattering Function
    qval = 2*np.pi/d.sigma*np.array([1,0])
    tval3, SelfInt2, SelfInt = allium.summstats.SelfIntermediate(d, qval,takeDrift,usetype=[1],verbose=plot)
    ss['tval3'] = tval3
    ss['SelfInt2'] = SelfInt2
    ss['SelfInt'] = SelfInt

    step = 10
    qmax = np.pi/d.sigma #particle size in wavelength (upper limit)
    # qmax *= 0.5 #interested in lower q values
    dx =  d.sigma#*0.5
    xmax = d.param.Ly*d.sigma
    ss['dx'] = dx
    ss['xmax'] = xmax

    # # E - real space velocity correlation function ('swirlyness')
    velcorrReal = np.zeros((800,))
    count = 0
    for u in range(0,endtime - starttime,step):
        # # # E - Real space velocity correlation function
        spacebins,velcorr = allium.summstats.getVelcorrSingle(d, dx,xmax,whichframe=u,usetype=usetypes,verbose=plot)
        velcorrReal[:len(spacebins)] += velcorr  

        count+=1

    velcorrReal/=count
    ss['velcorrReal'] = velcorrReal
    ss['spacebins'] = spacebins

    ssvect = [vav.mean(),
          stats.kurtosis(vdist,fisher=False),vdist.mean(),vdist.var(),\
          stats.kurtosis(vdist2,fisher=False),vdist2.mean(),vdist2.var(),\
          np.polyfit(np.log(tval[1:]), np.log(msd[1:]), 1)[0], \
          tval3[SelfInt2 < 0.5][0],\
          tval2[velauto < 1e-1][0],\
          np.polyfit(np.log(spacebins[50:400]), np.log(velcorrReal[50:400]), 1)[0]
          ] 
    return ss

def init_prior(bounds ,num_dim = 3): 
    """
    Returns prior (currently 3-dimensional parameter space sampled w uniform)
    """
    prior_min = bounds[0]
    prior_max = bounds[1]
    return utils.torchutils.BoxUniform(low=torch.as_tensor(prior_min),high=torch.as_tensor(prior_max))

def simulation_wrapper(params, test=False, log = False):
    """
    Returns summary statistics from active particle model of cells.

    Summarizes the output of the simulator and converts it to `torch.Tensor`.
    """
    thetafilename = f'output/v0_{int(params[0])}_k_{int(params[1])}_tau_{int(params[2])}'    
    if test:
        theta = [64, 24, 5]
        file = f'v0_{theta[0]:g}_k_{theta[1]:g}_tau_{theta[2]:g}.p'
        with open( file, 'rb') as f:
            obs = pickle.load(f)

    else:    
        obs = allium.simulate.sim(params, log)
    
    summstats = torch.as_tensor(calculate_summary_statistics(obs,log))

    save = random.uniform(0,1) < 0.01

    if save and not test:
        with open(thetafilename + '.p','wb') as f:
            pickle.dump(obs, f)

        with open(thetafilename + '_ss.p','wb') as f:
            pickle.dump(summstats, f)

    return summstats

def main():
    print('beginning run')
    #v0, k, tau = [30,150], [20,150], [1,10] <- parameter bounds

    tic = time.perf_counter() # <- time keeping

    #prior object must have sample attribute
    prior = init_prior([[30,20,1],[150,150,10]]) # <- see higher definition
    simulator, prior = prepare_for_sbi(simulation_wrapper, prior)

    flow_density_estimator_build_fun = posterior_nn(model='maf', hidden_features=60, num_transforms=3)
    mix_inference = SNPE( prior, density_estimator='mdn')
    flow_inference = SNPE( prior, density_estimator=flow_density_estimator_build_fun)
    
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=1, num_workers = 1)
    
    flow_density_estimator = flow_inference.append_simulations(theta, x).train()
    mix_density_estimator = mix_inference.append_simulations(theta, x).train()
    
    flow_posterior= flow_inference.build_posterior(flow_density_estimator) 
    mix_posterior = mix_inference.build_posterior(mix_density_estimator)  
    
    toc = time.perf_counter()
    print(f"Completed in {toc - tic:0.4f} seconds")

    picklefile = open('mixposterior.p', 'wb') 
    pickle.dump(mix_posterior, picklefile)
    picklefile = open('flowposterior.p', 'wb') 
    pickle.dump(flow_posterior, picklefile)

    print(f"Completed in {toc - tic:0.4f} seconds")

    return 0

if __name__ == "__main__":
    main()
