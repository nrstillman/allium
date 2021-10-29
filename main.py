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


def calculate_summary_statistics(d, log=False,starttime=60,endtime=320,takeDrift=False, plot = False):
    """
    Calculates summary statistics.

    """
    # 0 is new cells, 1 is tracer, 2 is original (check this)
    usetypes = [0,1,2]
    end = int(d.param.zaptime/d.param.output_time) #320
    # remove any data post zap
    d.truncateto(starttime, endtime)
    ssdata = {}
    # # # # # A - Velocity distributions and mean velocity
    velbins=np.linspace(0,10,100)
    velbins2=np.linspace(-10,10,100)
    vav, vdist,vdist2 = allium.summstats.getVelDist(d, velbins,velbins2, usetype=usetypes,verbose=plot)
    ssdata['vav'] = vav
    ssdata['vdist'] = vdist
    ssdata['vdist2'] = vdist2
    print('finished calculating A')
    # # B - Autocorrelation Velocity Function
    tval2, velauto, v2av = allium.summstats.getVelAuto(d, usetype=[1],verbose=plot)
    ssdata['tval2'] = tval2
    ssdata['velauto'] = velauto
    ssdata['v2av'] = v2av
    print('finished calculating B')
    # C - Mean square displacement
    tval, msd, d = allium.summstats.getMSD(d,takeDrift, usetype=[1],verbose=plot)
    ssdata['tval'] = tval
    ssdata['msd'] = msd
    print('finished calculating C')
    # # D - Self Intermediate Scattering Function
    qval = 2*np.pi/d.sigma*np.array([1,0])
    tval3, SelfInt2, SelfInt = allium.summstats.SelfIntermediate(d, qval,takeDrift,usetype=[1],verbose=plot)
    ssdata['tval3'] = tval3
    ssdata['SelfInt2'] = SelfInt2
    ssdata['SelfInt'] = SelfInt

    step = 10
    qmax = np.pi/d.sigma #particle size in wavelength (upper limit)
    dx =  d.sigma*0.9
    xmax = d.param.Ly
    ssdata['dx'] = dx
    ssdata['xmax'] = xmax
    print('finished calculating D')
    # # E - real space velocity correlation function ('swirlyness')
    velcorrReal = np.zeros((100,))
    count = 0
    for u in range(0,endtime - starttime,step):
        # # # E - Real space velocity correlation function
        spacebins,velcorr = allium.summstats.getVelcorrSingle(d, dx,xmax,whichframe=u,usetype=usetypes,verbose=plot)
        velcorrReal[:len(spacebins)] += velcorr  
        count+=1

    velcorrReal = velcorrReal[:len(spacebins)]
    velcorrReal/=count
    ssdata['velcorrReal'] = velcorrReal
    ssdata['spacebins'] = spacebins

    x = spacebins[(50<spacebins) & (spacebins < 300)]
    y = velcorrReal[(50<spacebins) & (spacebins< 300)]
    
    print('finished calculating E')

    ssvect = [vav.mean(),
          stats.kurtosis(vdist,fisher=False),vdist.mean(),vdist.var(),\
          stats.kurtosis(vdist2,fisher=False),vdist2.mean(),vdist2.var(),\
          np.polyfit(np.log(tval[1:]), np.log(msd[1:]), 1)[0], \
          tval3[SelfInt2 < 0.5][0],\
          tval2[velauto < 1e-1][0],\
          np.polyfit(np.log(x[y>0]), np.log(y[y>0]), 1)[0]
          ]
    print('finished calculating summ stats')
    return ssvect, ssdata

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
    thetafilename = f'output/v0_{int(params[0])}_k_{int(params[1])}_tau_{int(params[2])}_a_{int(params[3])}'    
    if test:
        theta = [130, 85, 7, 0]
        file = f'test_output/v0_{theta[0]:g}_k_{theta[1]:g}_tau_{theta[2]:g}_a_{theta[3]:g}.p'
        with open( file, 'rb') as f:
            obs = pickle.load(f)
    else:    
        obs = allium.simulate.sim(params, log)
    
    save = random.uniform(0,1) < 0.02
    
    if save and not test:
        with open(thetafilename + '.p','wb') as f:
            pickle.dump(obs, f)

    ssvect, ssdata = calculate_summary_statistics(obs,log)
    if save and not test:
        with open(thetafilename + '_ss.p','wb') as f:
            pickle.dump(ssdata, f)

    return torch.as_tensor(ssvect)

def main():
    print('beginning run')
    #v0, k, tau = [30,150], [20,150], [1,10], [1,10], [0.0,1] <- parameter bounds
    # additional:
    #a = [4e-4, 8e-3] (reflects a/d0 of [0.01 to 0.2])
    # check before including whether this is with sqrt(dt) and realistic ranges
    #align = []
    tic = time.perf_counter() # <- time keeping

    #prior object must have sample attribute
    # prior = init_prior([[30,20,1],[150,150,10]]) # <- see higher definition
    #additional
    prior = init_prior([[30,20,1, 4e-4],[150,150,10,8e-3]]) # <- see higher definition
    simulator, prior = prepare_for_sbi(simulation_wrapper, prior)

    flow_density_estimator_build_fun = posterior_nn(model='maf', hidden_features=60, num_transforms=3)
    mix_inference = SNPE( prior, density_estimator='mdn')
    flow_inference = SNPE( prior, density_estimator=flow_density_estimator_build_fun)
    
    theta, x = simulate_for_sbi(simulator, proposal=prior, num_simulations=1000, num_workers = 16)
    
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
