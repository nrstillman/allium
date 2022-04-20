import sys
import time
import copy 
import signal
# This will be replaced by a called function in later releases
sys.path.append('simulator/') 
import json
import numpy as np
import pickle 
import random
import torch

import pycapmd as capmd
import allium
from joblib import Parallel, delayed
from tqdm import tqdm

class Sim(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.counter = 1
        try:
            print(f'Parameter file loaded from {self.parameterFile}')
        except:
            self.parameterFile = "include/config/simconfig.json"
        #setting default values for testing simulation runs only 
        if not hasattr(self,'params'):
            self.params = ['v0', 'k', 'tau']
        else:
            self.params = [p[1] for p in self.pmap.items()]        
        if not hasattr(self,'pmap'):
            print('no parameter map detected, using default')
            self.pmap = {}
        if not hasattr(self,'log'):
            self.log = False
        if not hasattr(self,'nfeatures'):
            self.nfeatures = 15
        if not hasattr(self,'framerate'):
            self.framerate = 10/60
        if not hasattr(self,'test'):
            self.test = False
        if not hasattr(self,'pmap'):
            self.keys = ['factive', 'pairstiff', 'tau', 'deathrate']    
        else:
            self.keys = list(self.pmap.keys())
        
        if self.test:
            print(self.keys)
            if not hasattr(self,'test_theta'):
                if len(self.keys) == 3:
                    self.test_theta = [130, 85, 7]
                if len(self.keys) == 6:
                    self.test_theta = [6.0225e+01, 7.4157e+01, 4.0066e+00, 4.4993e-01, 2.3977e-03, 9.9745e+02]
                else:
                    print("ERROR: No default parameters saved for this number of parameters. Set test parameters with test_theta")

    def sample(self):
        """
        Sample simulator from proposed prior 

        """
        print(self.num_simulations)
        x = torch.Size([self.num_simulations])
        if len(self.starttime) > 1:
            print("Caution: Running scratch and confluent simultaneously")

        theta = self.proposal.sample(sample_shape=torch.Size([self.num_simulations]))    
        self.thetafilename = self.folder + self.run + '_'
        print(theta, file=open(f'{self.thetafilename}sampled_theta_.txt', 'a'))
        batches = torch.split(theta, self.batch_size, dim=0)
        
        with allium.utils.tqdm_joblib(tqdm(desc="Running simulations", total=self.batch_size)) as progress_bar:
            simulation_outputs = Parallel(n_jobs=self.num_workers)(
                    delayed(self.wrapper)(batch) for batch in batches)

        x = torch.cat(simulation_outputs, dim=0)
        print(x.shape)
        return theta, simulation_outputs
        
    def wrapper(self, p):
        """
        Returns summary statistics from active particle model of cells.

        Summarizes the output of the simulator and converts it to `torch.Tensor`.
        """
        idx = []
        xout = torch.Tensor()#zeros((1,15,1))
        for i, params in enumerate(p):
            print(f'\nRunning simulation {self.counter}/{self.num_simulations} w params {params}', file=open(f'{self.folder}_log.txt', 'a'))
            def sig_handler(signum, frame):
                print(f'Error: segfault w params {params}')

            signal.signal(signal.SIGSEGV, sig_handler)

            self.thetafilename = self.folder + self.run + '_'
            for (a,b) in zip(list(self.pmap.keys()),params):
                self.thetafilename+=f'{a}_{b:.1e}_'
            if self.test:
                theta = self.test_theta
                try:
                    testout = self.test_folder
                except:
                    testout = 'test_output'
                file = f'{self.run}'
                for (p,t) in zip(self.params,theta):
                    file+= f'_{p}_{t}'

                file = f'{testout}/{file}.p'
                with open( file, 'rb') as f:
                    obs = pickle.load(f)
            else:    
                obs = self.simulate(params)
                if obs == None:
                    return torch.as_tensor([0]*self.nfeatures)

            save = random.uniform(0,1) < self.save_prob
            
            if save and not self.test:
                with open(self.thetafilename[:-1] + '.p','wb') as f:
                    pickle.dump(obs, f)
            #Calculate summary statistics here
            try:
                if len(self.starttime) > 1:
                    ssvect =[]
                    ssdata =[]
                    for s,e in zip(self.starttime, self.endtime):
                        tmp_obs = copy.deepcopy(obs)
                        tmp_obs.param.framerate = self.framerate
                        # rescale time based on frame rate
                        vect0, data0 = allium.summstats.calculate_summary_statistics(tmp_obs,useall = self.useall,opts = self.ssopts,log = self.log, starttime=s, endtime=e,usetypes=[1,2],log_output=f'{self.folder}_log.txt')
                        ssvect.append(vect0)
                        ssdata.append(data0)
                        #save with starttime            
                        with open(f'{self.thetafilename}_starttime_{s}_ss.p','wb') as f:
                            pickle.dump([ssvect, ssdata, obs.param], f)
                    # below is horrible... needs to be sorted
                    ssvect = torch.as_tensor(np.asarray((ssvect[0],ssvect[1])).reshape(1,len(ssvect[0]),2))
                    xout = torch.cat((xout, ssvect),0)                
                else:
                    obs.param.framerate = self.framerate
                    ssvect, ssdata = allium.summstats.calculate_summary_statistics(obs,useall = self.useall,opts = self.ssopts,log = self.log, starttime=self.starttime[0], endtime=self.endtime[0],usetypes=[1,2],log_output=f'{self.folder}_log.txt')
                    #save with starttime            
                    with open(f'{self.thetafilename}_starttime_{self.starttime[0]}_ss.p','wb') as f:
                        pickle.dump([ssvect, ssdata, obs.param], f)
                    ssvect = torch.as_tensor(np.asarray(ssvect).reshape(1,len(ssvect)))    
                    xout = torch.cat((xout, ssvect),0)                
            except Exception as e:
                bad_output = f'{self.thetafilename}_badss.p'
                print(f"Error: Exception raised during calculation of summary statistiscs. Output saved to {bad_output}")
                print(e)
                with open(bad_output,'wb') as f:
                    pickle.dump(obs, f)
                pass          
            idx.append(i)
        return torch.as_tensor(xout)#, p[idx]

    def simulate(self, params):
        """
        Main simulation function

        """
        def printOutput(t, tic, p, log=False):
            toc = time.perf_counter()
            message =  f"""-----------------------
                        \n Timestep: {t} \
                        \n # of Cells: {len(p)} \
                        \n Since last log {toc - tic[1]:.4} seconds \
                        \n Total Runtime: {toc - tic[0]:.4} seconds
                        \n -----------------------"""
            if log:
                print(f'{message}', file=open(f'{self.folder}_log.txt', 'a'))
            else:
                print(f'{message}')        
                print("\033[9A")
            return 0

        def getData(sim, neighbours = False):
            popidx = [i for i in range(sim.popSize())]
            pop = sim.getPopulationId(capmd.VectorInt(popidx))
            data = []
            for p in list(pop): 
                if sim.getParticle(p).getType() == 1:
                    data.append([sim.getParticle(p).getId(),sim.getParticle(p).getType(),sim.getParticle(p).getVelocity()])

            popId = np.array(sim.getPopulationId(capmd.VectorInt(popidx))).reshape(len(popidx),1)
            popPosn = np.array(sim.getPopulationPosition(capmd.VectorInt(popidx)))
            popType = np.array(sim.getPopulationType(capmd.VectorInt(popidx))).reshape(len(popidx),1)
            popArray = np.append(popId,popPosn,axis=1)
            popArray = np.append(popArray,popType, axis=1)
            if neighbours:
                popPosn = np.array(sim.getPopulationPosition(capmd.VectorInt(popidx)))
                popType = np.array(sim.getPopulationType(capmd.VectorInt(popidx))).reshape(len(popidx),1)
                
            return popArray

        def getPopulation(sim, neighbours = False):
            pop = sim.popSize()
            popidx = []
            for i in range(pop): 
                popidx.append(i)

            popId = np.array(sim.getPopulationId(capmd.VectorInt(popidx)))
            popPosn = np.array(sim.getPopulationPosition(capmd.VectorInt(popidx)))
            popVel = np.array(sim.getPopulationVelocity(capmd.VectorInt(popidx)))
            popTheta = np.array(sim.getPopulationTheta(capmd.VectorInt(popidx)))
            popRadius = np.array(sim.getPopulationRadius(capmd.VectorInt(popidx)))
            popType = np.array(sim.getPopulationType(capmd.VectorInt(popidx)))
            popArray = np.stack([popId,popPosn[:,0], popPosn[:,1],
                                 popVel[:,0], popVel[:,1],
                                 popTheta, popRadius ,popType], axis=1)

            if neighbours:
                print('Error: currently not implemented')
                #get neighbours here
            return popArray

        def updateParams(p, params,keys,tracers=0.1, log=False):
            setattr(params, 'log', log)
            if not bool(len(p)):
                print("No parameters updated")
            else:
                for key, value in zip(keys[:len(p)],p):  
                    print(key,value)                                      
                    value = np.array(value)
                    if log:
                        print(f'{key} = {value}\n', file=open(f'{self.folder}_log.txt', 'a'))
                    else:
                        print(f'{key} = {value}\n')
                    if key == 'pairstiff':
                        setattr(params, key, [[value,value,value],[value,value,value],[value,value,value]])
                    elif key == 'pairatt':
                        setattr(params, key, [[value,value,value],[value,value,value],[value,value,value]])
                    elif key == 'N':
                        setattr(params, 'N', int(value))
                        setattr(params, 'Ntracer', int(value*tracers))
                    elif key == 'phi':
                        setattr(params, 'phi', value)
                        # setattr(params, 'Ntracer', int(value*tracers))
                    elif (key == 'deathrate') or (key == 'divrate'):
                        setattr(params, key, [value,0,value])
                    else:
                        setattr(params, key, [value,value,value])
            return params

        def paramsFromFile(paramObj, fileName):
            paramObjCopy = paramObj
            try:
                with open(fileName) as jsonFile:
                    parameters = json.load(jsonFile)
                    for attribute in parameters:
                        setattr(paramObjCopy, attribute, parameters[attribute])
                    return paramObjCopy
            except Exception as e:
                print(e)
                print("Incorrect json format, returning default parameters")
                return paramObj
    
        if self.log:
            print(f"# of parameters = {len(params)}", file=open(f'{self.folder}_log.txt', 'a'))
        else:
            print(f"# of parameters = {len(params)}")
        tic = time.perf_counter()
        tic2 = time.perf_counter()
        defaultparams = paramsFromFile(capmd.Parameters(), self.parameterFile)
        params = updateParams(params, defaultparams,self.keys,log=self.log)
        sim = capmd.interface(params)
        timesteps = []
        x = []
        Rlength = params.Lx/4
        maxR = [ Rlength/2,  params.Ly]
        minR = [-Rlength/2, -params.Ly]
        popArray = []
        for t in range(params.t_final):
            sim.move()
            # Test for output
            if (t % params.output_time == 0): 
                p = getPopulation(sim)   
                printOutput(t, [tic, tic2], p,self.log)
                popArray.append(p)         
                if (params.output_type == 'all'):
                    sim.saveData("text")
                    sim.saveData("vtp")
                else:
                    sim.saveData(params.output_type)            
                tic2 = time.perf_counter()
                timesteps.append(t)

            # Test for scratch
            if (t == params.zaptime):
                p = getPopulation(sim)            
                zapList = []
                for i in range(sim.popSize()):
                    x = sim.getPopulationPosition(capmd.VectorInt([i]))[0]
                    if ((x[0] < maxR[0]) & (x[0] > minR[0])):
                        if ((x[1] < maxR[1]) & (x[1] > minR[1])):
                            idx = sim.getPopulationId(capmd.VectorInt([i]))[0]
                            zapList.append(idx)
                
                sim.killCells(capmd.VectorInt(zapList))
                if self.log:
                    print("\n"*10+"Cell zapping stage completed" + "\n"*2,end="", file=open(f'{self.folder}_log.txt', 'a'))
            # Test for population dynamicss
            if (t % params.popdynfreq == 0): 
                sim.populationDynamics(params.popdynfreq)
            
        d = {}
        for att in dir(params):
            if not(att.startswith('__')):
                d[att] =  getattr(params,att)
        self.counter +=1
        return allium.data.SimData(params=d, data=popArray, loadtimes = [0,int(params.t_final/params.output_time)])

