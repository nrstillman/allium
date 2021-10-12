# script for running sbi without simulations
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os 

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
            return [sim['data'][ts] for ts in range(final_time)]

def dataloader(sim_x = [], sim_theta = [], path = 'output/', final_time = 420, 
                nfiles = 10, summstats = True, tracers = False):

    print(f'\t summstats = {summstats},\n\t tracers = {tracers},\n\t number of files = {nfiles}')

    def new_summarystatistic(data):
            # fake new summary statistic which calculates the time averaged msd for 3 tracers particles 
            def get_trajs(idx,data):
                # gets trajectory of a particle
                traj = []
                for d in data:
                    traj.append(d[d[:, 0] == idx])
                return np.array(traj).reshape(final_time,4)

            ss1 = np.linalg.norm(get_trajs(3900,data)[:,1:3], axis=1).mean()
            ss2 = np.linalg.norm(get_trajs(3950,data)[:,1:3], axis=1).mean()
            ss3 = np.linalg.norm(get_trajs(3990,data)[:,1:3], axis=1).mean()

            return torch.as_tensor([ss1,ss2])

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
                x = new_summarystatistic(x)
                sim_x.append(x.float())

            sim_theta.append(theta)
            counter +=1
            print(f'loaded {counter}/{nfiles} ({100*counter/nfiles}%)', end = '\r')

        print('\nFinished loading simulator output & parameters')
    else:
        print('\nNothing loaded. Using pre-loaded simulator output & parameters\n')
    
    return sim_x, sim_theta


def plot_posterior(posterior, x_o, points):

    posterior_samples = posterior.sample((10000,), x=x_o)

    # plot posterior samples
    _ = analysis.pairplot(posterior_samples, limits=[[30,150],[20,150],[1,10]], 
                        figsize=(5,5), labels=['v0', 'k', 'tau'], 
                        points = points,points_colors = 'r')
    plt.show()
    return 0 

def main(post_file = ''):

    path = 'output/'

    # Number of output files to use
    nfiles = 239
    # whether to use preloaded summary statistics (ss) or trajectories
    summstats = True
    # whether to use all trajectories or just tracer particles
    tracers = True
    # Number of timesteps to use (zap occurs at 319) - only used if summstats = False
    final_time = 320

    if tracers and final_time > 320: print('NOTE! zap at 320 means tracers may be lost and referencing doesnt work')
    
    #observed summary statistic for theta = [98, 97, 7] <- used for testing posterior
    point = [[98,97,7]]
    x_o = 5.5062e-01,  5.0076e+01, -5.6172e-03

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
    p = ''#'confluentpost.p'
    main(post_file = p)