import sys
# This will be replaced by a called function in later releases
sys.path.append('simulator/') 

import json
import time
import pickle 
import random
import torch

import numpy as np
import pycapmd as capmd
import allium

class Simulate(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        try:
            print(f'Parameter file loaded from {self.parameterFile}')
        except:
            self.parameterFile = "include/config/simconfig.json"

        self.params = [p[1] for p in self.pmap.items()]
        self.keys = list(self.pmap.keys())
        if self.test:
            try:
                print(f'test theta is {self.test_theta}')
            except ValueError:
                if len(self.keys) == 3:
                    self.test_theta = [130, 85, 7]
                else:
                    print("No default parameters saved for this number of parameters. Set test parameters with test_theta")
                    return 1 

    def wrapper(self, params):
        """
        Returns summary statistics from active particle model of cells.

        Summarizes the output of the simulator and converts it to `torch.Tensor`.
        """

        thetafilename = f'{self.folder}/v0_{int(params[0])}_k_{int(params[1])}_tau_{int(params[2])}'    
        if self.test:
            theta = self.test_theta
            try:
                testout = self.test_folder
            except:
                testout = 'test_output'
            file = 'p'
            for (p,t) in zip(self.params,theta):
                file+= f'_{p}_{t}'

            file = f'{testout}/{file}.p'
            with open( file, 'rb') as f:
                obs = pickle.load(f)
        else:    
            obs = sim(params, keys = self.keys, log = self.log)
            print(params)
            obs = params
        save = random.uniform(0,1) < self.save_prob
        
        if save and not self.test:
            with open(thetafilename + '.p','wb') as f:
                pickle.dump(obs, f)

        ssvect, ssdata = allium.summstats.calculate_summary_statistics(obs,opts = self.opts,log = self.log, starttime=self.starttime, endtime=self.endtime)

        if save and not self.test:
            with open(thetafilename + '_ss.p','wb') as f:
                pickle.dump(ssdata, f)

        return torch.as_tensor(ssvect)

    def sim(self):
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
                print(f'{message}', file=open('log.txt', 'a'))
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

        def updateParams(p, params,keys,log=False):
            setattr(params, 'log', log)
            if not bool(len(p)):
                print("No parameters updated")
            else:
                for key, value in zip(keys,p):
                    value = np.array(value)
                    if log:
                        print(f'{key} = {value}\n', file=open('log.txt', 'a'))
                    else:
                        print(f'{key} = {value}\n')
                    if key == 'pairstiff':
                        setattr(params, key, [[value,value,value],[value,value,value],[value,value,value]])
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
            print(f"# of parameters = {len(p)}", file=open('log.txt', 'a'))
        else:
            print(f"# of parameters = {len(p)}")
        tic = time.perf_counter()
        tic2 = time.perf_counter()
        params = paramsFromFile(capmd.Parameters(), self.parameterFile)
        params = updateParams(p, self.params,self.keys,self.log)
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
                printOutput(t, [tic, tic2], p,log)
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
                print("\n"*10+"Cell zapping stage completed" + "\n"*2,end="")
            # Test for population dynamicss
            if (t % params.popdynfreq == 0): 
                sim.populationDynamics(params.popdynfreq)
            
        d = {}
        for att in dir(params):
            if not(att.startswith('__')):
                d[att] =  getattr(params,att)
            
        return allium.data.SimData(params=d, data=popArray, loadtimes = [0,int(params.t_final/params.output_time)])

    def sim_neighbours(self):
        if log:
            print(f"# of parameters = {len(p)}", file=open('log.txt', 'a'))
        else:
            print(f"# of parameters = {len(p)}")
        tic = time.perf_counter()
        tic2 = time.perf_counter()
        parameterFile = "include/config/simconfig_neighbours.json"
        params = paramsFromFile(capmd.Parameters(), parameterFile)
        params = updateParams(p, params,log)
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
                printOutput(t, [tic, tic2], p,log)
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
                print("\n"*10+"Cell zapping stage completed" + "\n"*2,end="")
            # Test for population dynamicss
            if (t % params.popdynfreq == 0): 
                sim.populationDynamics(params.popdynfreq)
            
        d = {}
        for att in dir(params):
            if not(att.startswith('__')):
                d[att] =  getattr(params,att)

        return {'data':popArray, 'params':d, 'time':timesteps, 'dt':timesteps[1] - timesteps[0]}

