# script for running sbi without simulations
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os 

import allium
from scipy import stats

#special numpy for neural networks
import torch
#neural network architecture for inference
from sbi.inference import SNPE
#only used to make pairplots
from sbi import analysis

def simulatorloader(theta, final_time = 480, path = 'output/', summstats = True, tracers = False):
    file = f'v0_{theta[0]:g}_k_{theta[1]:g}_tau_{theta[2]:g}.p'
    with open(path + file, 'rb') as f:
        sim = pickle.load(f)

    if summstats:
        return sim['ss']

    else:
        if tracers:
            # gets all tracer particles and returns array shaped as 
            tracers = sim['data'][0][sim['data'][0][:,-1] == 1.][:,0]
            new_data = []
            for ts in range(final_time):                
                # rebuild dataset keeping only tracer particles
                xdata = []
                for x in tracers:
                    xdata.append(sim['data'][ts][sim['data'][ts][:,0] == x])
                new_data.append(np.array(xdata).reshape(len(xdata),4))
            
            return new_data
        else:
            return sim

def dataloader(sim_x = [], sim_theta = [], path = 'output/', final_time = 420, 
                nfiles = 10, summstats = True, tracers = False):

    print(f'\t summstats = {summstats},\n\t tracers = {tracers},\n\t number of files = {nfiles}')

    def new_summarystatistic(d,theta,starttime=60, endtime=320,takeDrift=False,plot=False):
        """
        Calculates summary statistics.

        """
                
        takeDrift = False #< check this
        plot = False
        starttime = 60
        endtime = 320
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
        return torch.as_tensor(ss)

    # get all output files in path
    runs = os.listdir(path)

    counter = 1
    # check if sim_theta/sim_x are already loaded
    if not bool(len(sim_x)):
        sim_theta = []
        sim_x = []
        for f in runs[:nfiles]:
            # skip any non-pickle files
            if not f.endswith('.p'):
                continue

            # load saved output parameters
            theta = [float(f.split('_')[1]),\
                     float(f.split('_')[3]),\
                     float(f.split('_')[5].split('.')[0])]  

            x = simulatorloader(theta, final_time = final_time, \
                            summstats = summstats, tracers = tracers)
            if summstats:
                #output from simulatorloader is summary statistics 
                sim_x.append(x.float())
            else:
                #output from simulatorloader is all trajectories
                # calculate new summary statistics using function above
                x = new_summarystatistic(x,theta)
                sim_x.append(x.float())

            sim_theta.append(theta)
            counter +=1
            print(f'loaded {counter}/{nfiles} ({100*counter/nfiles}%)', end = '\r')

        print('\nFinished loading simulator output & parameters')
    else:
        print('\nNothing loaded. Using pre-loaded simulator output & parameters\n')
    
    return sim_x, sim_theta


def plot_posterior(posterior, x_o, points):

    posterior_samples = posterior.sample((1000000,), x=x_o)

    # plot posterior samples
    _ = analysis.pairplot(posterior_samples, limits=[[30,150],[20,50],[1,10]], 
                        figsize=(5,5), labels=['v0', 'k', 'tau'], 
                        points = points,points_colors = 'r')
    plt.show()
    return 0 

def main(post_file = ''):

    path = 'output/'

    # Number of output files to use
    nfiles = 22
    # whether to use preloaded summary statistics (ss) or trajectories
    summstats = False
    # whether to use all trajectories or just tracer particles
    tracers = False
    # Number of timesteps to use (zap occurs at 319) - only used if summstats = False
    final_time = 320

    if tracers and final_time > 320: print('NOTE! scratch at 320 means tracers may be lost and referencing doesnt work')
    
    #observed summary statistic for theta=
    point = [[98,71,4]]
    x_o  = [8.610529135132671, 7.89438968534453, 0.09999999999999999, \
            0.04974817367575438, 11.821959604501034, 0.05, 0.018131065407073408, \
            1.4147876299837543, 0.49992277992278, 0.99984555984556]
    if len(post_file) > 0:     
        with open('old_posteriors/' + post_file, 'rb') as f:
            posterior = pickle.load(f)
    else:
        #Load presimulated data (saved in path)
        print('Loading data')
        sim_x, sim_theta = dataloader(summstats = summstats, tracers = tracers, nfiles = nfiles, final_time = final_time)

        # Setup the inference procedure using a sequential neural posterior estimator SNPE
        # The neural network here is a mixture density network (mdn) which combines gaussians, 
        # also possible to try a type of normalising flow - masked autoregressive flow (maf)
        inference = SNPE(prior=None, density_estimator = 'maf')

        # Recast theta and x as appropriate types for the estimator
        theta = torch.as_tensor(sim_theta)
        x = torch.stack(sim_x)

        #Append data to the inference engine (defined above) and train
        print('\nTraining neural density estimator')
        density_estimator = inference.append_simulations(theta, x).train()

        #Build the posterior using the neural network 
        posterior = inference.build_posterior(density_estimator)

    #Plot the posterior, highlighting an example summary statistic (x_o) and parameter combination (point)
    plot_posterior(posterior, x_o, points = point )
    return 0 

if __name__ == "__main__":
    p = ''#'mixposterior.p'
    main(post_file = p)