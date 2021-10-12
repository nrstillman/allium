import numpy as np
import pandas as pd
import glob
import json

import btrack
import napari #prefer to drop this in the future

def load_exp(filenumber = 1, folder = '../cell_track/segmenting/slow/*.csv', configfile = '../cell_track/cell_config.json'):
    files = glob.glob(folder)
    if len(files) == 0:
        return('bad folder dir')
    else:
        file = files[filenumber]
    print(file)
    data =  pd.read_csv(file)
    segments = data[['ImageNumber',
                    'Location_Center_X',
                    'Location_Center_Y']].to_csv('tmp.csv', header = ['t', 'x', 'y'])
 
    objects = btrack.dataio.import_CSV('tmp.csv')
    with btrack.BayesianTracker() as tracker:

        # configure the tracker using a config file
        tracker.configure_from_file(configfile)

        tracker.append(objects)
        tracker.volume=((0,1000), (0,1000), (-1e5,1e5))

        # track and optimize
        tracker.track_interactive(step_size=10)
        tracker.optimize()

        # get the tracks in a format for napari visualization
        data, properties, graph = tracker.to_napari(ndim=2)

    output = pd.DataFrame(data,columns = ['flag', 'time', 'x','y'])
    data = []
    for t in output.time.unique():
        x = output[output['time'] == t].x.values
        y = output[output['time'] == t].y.values
        flags = output[output['time'] == t].flag.values
        tmp = np.append(flags.reshape(len(x),1),x.reshape(len(x),1), axis=1)
        data.append(np.append(tmp,y.reshape(len(x),1),axis=1))

    tracers = []
    for n in data[0][:,0]:
        if n in data[-1][:,0]:     
            tracers.append(n)

    for i, d in enumerate(data):
        celltype = np.in1d(d[:,0],tracers)*1
        data[i] = np.append(d,celltype.reshape(len(d),1), axis=1)

    return data, output

def read_output(f):
    build = True
    for line in open(f, 'r'):
        item = line.rstrip()
        if build:
            col = item.split(',')
            data = pd.DataFrame(columns = col)
            build = False
        else:
            row = pd.Series(item.split(','), index=col)
            data = data.append(row, ignore_index = True)
    return data

def read_params(configfile):
    params = dict()
    with open(configfile) as jsonFile:
        parameters = json.load(jsonFile)
        for attribute in parameters:
            params[attribute] = parameters[attribute]
    return parameters

def load_sim(folder ='.data/sim/testdata', configfile=  ".data/config/simconfig.json"):
    params=read_params(configfile)
    states= []
    t = []
    print('\nLoading simulation data\n')
    for f in sorted(glob.glob(folder + '*.dat')):
        print(f +  '\r', end = "")
        states.append(read_output(f))
        t.append(f.split('/')[-1].split('_')[1][:-4])

    dt = int(t[1]) - int(t[0])
    
    return dict(data=states, time=t, dt=dt, params=params)
        