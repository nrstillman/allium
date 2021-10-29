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
