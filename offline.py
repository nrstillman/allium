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


def plot_posterior(posterior, x_o, points):

    posterior_samples = posterior.sample((1000000,), x=x_o)

    if len(points[0]) == 3:
        print('3params')
        limits = [[30,150],[20,150],[1,10]]
        labels = ['v0', 'k', 'tau']
    elif len(points[0]) == 4:
        print('4params')
        limits = [[30,150],[20,150],[1,10], [4e-4,8e-3]]
        labels = ['v0', 'k', 'tau', 'a']

    # plot posterior samples
    _ = analysis.pairplot(posterior_samples, limits=limits, 
                        figsize=(5,5), labels=labels, 
                        points = points,points_colors = 'r',
                        title=f'theta = {points[0]} (mdn)')
    plt.show()
    return 0 

def simulatorloader(theta, final_time = 480, path = 'output/', summstats = True, tracers = False):
    file = f'v0_{theta[0]:g}_k_{theta[1]:g}_tau_{theta[2]:g}.p'
    with open(path + file, 'rb') as f:
        sim = pickle.load(f)

    theta = [sim.param.factive[0],sim.param.pairstiff[0][0],sim.param.tau[0]]
    if summstats:
        with open(path + file[:-2] + '_ss.p', 'rb') as f:
                ss = pickle.load(f)

        x = ss['spacebins'][(50<ss['spacebins']) & (ss['spacebins'] < 250)]
        y = ss['velcorrReal'][(50<ss['spacebins']) & (ss['spacebins']< 250)]
        fit = np.polyfit(np.log(x[y>0]), np.log(y[y>0]), 1)[0]
        if np.isnan(fit):
            return print('Error: cannot fit exponent')
        ssvect = [ss['vav'].mean(),
          stats.kurtosis(ss['vdist'],fisher=False),ss['vdist'].mean(),ss['vdist'].var(),\
          stats.kurtosis(ss['vdist2'],fisher=False),ss['vdist2'].mean(),ss['vdist2'].var(),\
          np.polyfit(np.log(ss['tval'][1:]), np.log(ss['msd'][1:]), 1)[0], \
          ss['tval3'][ss['SelfInt2'] < 0.5][0],\
          ss['tval2'][ss['velauto'] < 1e-1][0],\
          fit
          ]
        return ssvect, theta
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
            
            return new_data,theta
        else:
            return sim, theta

def dataloader(sim_x = [], sim_theta = [], path = 'output/', final_time = 420, 
                nfiles = 10, summstats = True, tracers = False):

    print(f'\t summstats = {summstats},\n\t tracers = {tracers},\n\t number of files = {nfiles}')

    def new_summarystatistic(d,theta,starttime=321, endtime=480,takeDrift=False,plot=False):
        """
        Calculates summary statistics.

        """
        print('Calculating summ stats')
        # 0 is new cells, 1 is tracer, 2 is original (check this)
        usetypes = [0,1,2]
        end = int(d.param.zaptime/d.param.output_time) #320
        # remove any data post zap
        d.truncateto(starttime, endtime)
        ss = {}
        # A - Velocity distributions and mean velocity
        velbins=np.linspace(0,10,100)
        velbins2=np.linspace(-10,10,100)
        vav, vdist,vdist2 = allium.summstats.getVelDist(d, velbins,velbins2, usetype=usetypes,verbose=plot)
        # B - Autocorrelation Velocity Function
        tval2, velauto, v2av = allium.summstats.getVelAuto(d, usetype=[1],verbose=plot)
        # C - Mean square displacement
        tval, msd, d = allium.summstats.getMSD(d,takeDrift, usetype=[1],verbose=plot)
        # D - Self Intermediate Scattering Function
        qval = 2*np.pi/d.sigma*np.array([1,0])
        tval3, SelfInt2, SelfInt = allium.summstats.SelfIntermediate(d, qval,takeDrift,usetype=[1],verbose=plot)

        step = 10
        qmax = np.pi/d.sigma #particle size in wavelength (upper limit)
        dx =  d.sigma#*0.5
        xmax = d.param.Ly*d.sigma
        # E - real space velocity correlation function ('swirlyness')
        velcorrReal = np.zeros((800,))
        count = 0
        upperlimit = 150 #scratch
        upperlimit = 260
        print('Finished calculating the easy ones')
        for u in range(0,upperlimit,step):
            spacebins,velcorr = allium.summstats.getVelcorrSingle(d, dx,xmax,whichframe=u,usetype=usetypes,verbose=plot)
            velcorrReal[:len(spacebins)] += velcorr  

            count+=1
        velcorrReal = velcorrReal[:len(spacebins)]
        velcorrReal/=count
        x = spacebins[(50<spacebins) & (spacebins < 250)]
        y = velcorrReal[(50<spacebins) & (spacebins< 250)]
        print('Finished calculating summ stats')
        ssvect = [vav.mean(),
              stats.kurtosis(vdist,fisher=False),vdist.mean(),vdist.var(),\
              stats.kurtosis(vdist2,fisher=False),vdist2.mean(),vdist2.var(),\
              np.polyfit(np.log(tval[1:]), np.log(msd[1:]), 1)[0], \
              tval3[SelfInt2 < 0.5][0],\
              tval2[velauto < 1e-1][0],\
              np.polyfit(np.log(x), np.log(y), 1)[0]
              ]

        return [float(s) for s in  ssvect]

    # get all output files in path
    runs = os.listdir(path)

    counter = 0
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

            obs,theta = simulatorloader(theta, path=path, final_time = final_time, \
                            summstats = summstats, tracers = tracers)

            if summstats:
                #output from simulatorloader is summary statistics 
                sim_x.append(obs)
                # print(torch.tensor(obs))

            else:
                #output from simulatorloader is all trajectories
                # calculate new summary statistics using function above
                ss = new_summarystatistic(obs,theta)
                sim_x.append(ss)

            sim_theta.append(theta)
            counter +=1
            print(f'loaded {counter}/{nfiles} ({100*counter/nfiles}%)', end = '\r')

        print('\nFinished loading simulator output & parameters')
    else:
        print('\nNothing loaded. Using pre-loaded simulator output & parameters\n')
    
    return sim_x, sim_theta

def main(post_file = ''):

    path = 'output/3params/'

    # Number of output files to use
    nfiles = 50
    # whether to use preloaded summary statistics (ss) or trajectories
    summstats = True
    # whether to use all trajectories or just tracer particles
    tracers = False #<-- not used
    # Number of timesteps to use (zap occurs at 319) - only used if summstats = False
    final_time = 480

    if tracers and final_time > 320: print('NOTE! scratch at 320 means tracers may be lost and referencing doesnt work')
    
    #observed summary statistic for theta (v0, k, theta)
    # point = [[130.881,85.401,7.885,0.004]]
    # x_o  = [10.619862140890774, 7.281318119420386, 0.09999999999999999, 
    #         0.046286825048972746, 14.215249546296318, 0.05000000000000001, 
    #         0.018488439758237423, 1.6700859307454134, 0.33328185328185334, 
    #         6.165714285714286,  -1.3076073560018153]

    # point = [[106.495,76.78,5.22]]
    # x_o = [8.710863933824555, 7.006293753291603, 0.1,\
    #      0.04575688259407634, 12.897090326037159, 0.05, \
    #      0.017781969361675913, 1.5188451282881101, 0.41660231660231667,\
    #       2.0830115830115834, -1.0103268536979177]

    point = [[43.144,118.356,7.517]]
    x_o = [2.4005848910622487, 7.192558655266788, 0.09999999999999999, \
            0.04864900330486766, 13.807987463527866, 0.05, \
            0.019013933524078055, 1.3440457369978542, 2.332972972972973, \
            0.6665637065637067, -0.7472524105546924]

    if len(post_file) > 0:     
  
    else:
        #Load presimulated data (saved in path)
        print('Loading data')
        sim_x, sim_theta = dataloader(path = path,summstats = summstats, tracers = tracers, nfiles = nfiles, final_time = final_time)

        # Setup the inference procedure using a sequential neural posterior estimator SNPE
        # The neural network here is a mixture density network (mdn) which combines gaussians, 
        # also possible to try a type of normalising flow - masked autoregressive flow (maf)
        inference = SNPE(prior=None, density_estimator = 'maf')

        # Recast theta and x as appropriate types for the estimator
        theta = torch.as_tensor(sim_theta)
        x = torch.as_tensor(sim_x, dtype=torch.float32)

        #Append data to the inference engine (defined above) and train
        print('\nTraining neural density estimator')
        density_estimator = inference.append_simulations(theta, x).train()

        #Build the posterior using the neural network 
        posterior = inference.build_posterior(density_estimator)

    #Plot the posterior, highlighting an example summary statistic (x_o) and parameter combination (point)
    plot_posterior(posterior, x_o, points = point )
    return posterior, x_o, point

if __name__ == "__main__":
    p = '4params/flowposterior_4params.p'
    posterior, x_o, point = main(post_file = p)