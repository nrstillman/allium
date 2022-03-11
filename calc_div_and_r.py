import glob
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

def get_particle(n,data):
    return(data[data.iloc[:,0] == n])

def get_time(t,data):
    return data[data.iloc[:,1] == t]

def summary_statistics(data):
    tmax = [len(get_particle(n,data)) for n in data.iloc[:,0].unique()]
    Nt = [len(get_time(n,data)) for n in data.iloc[:,1].unique()]
    ss = {'Tmax' : len(tmax), 'Nt': len(data.iloc[:,1].unique()),\
        'flags': data.iloc[:,0].unique(), 'Ncells' : tmax, 'Ncells_T' : Nt}
    return ss

files =  glob.glob('segments/tracked_/*.csv')#glob.glob('./segmenting/fast/*.csv')

division = True
show = False
div_rate = []
mean_r = []
resolution = 0.8
framerate = 10/60

sample_range = 100*framerate
overlap = 25*framerate
div_dict = {}
N_all = []
div_all = []
labels = []

for f in files:
    print(f'\n\n{f}')
    N_div_data = []
    # if (f.split('/')[3][0] == 'c') and (division):
    if division:
        data= pd.read_csv(f)
        data['t'] = [int(l.split(':')[-1]) for l in data.Label]
        all_x = np.unique(data.t.values)*framerate
        all_y = data.groupby('t').count().values[:,0]

        start = all_x[0] 
        finish = all_x[-1]
        
        while start+sample_range< finish:
            print(f'interval = {int((start)/framerate),int((start+ sample_range)/framerate)}')    
            x = all_x[(all_x > start) & (all_x<start+sample_range)]
            y = all_y[(all_x > start) & (all_x<start+sample_range)]

            N0 = y[0]
            N_all.append(N0)
            Nmax = y[-1]
            print(f'N0 = {N0},Nmax = {Nmax}')
            
            #Slope calculation
            print(f"Slope = {np.polyfit(x, np.log(y), 1)[0]:.2}" )
            m = np.polyfit(x, np.log(y), 1)[0]      
            div_all.append(m)
            labels.append(f.split('/')[2][:4])
            
            if show:
                plt.figure(dpi=100)
                plt.plot(x, np.poly1d(np.polyfit(x, np.log(y), 1))(x), 'r')
                plt.scatter(x,np.log(y))
                # plt.title(f)
                plt.xlabel('Hours')
                plt.ylabel('log(Number of cells)')
                # plt.annotate(m, [15,np.log(y[36])], rotation=20)
                plt.savefig(f'{f}_{int((start)/framerate)}_{int((start + sample_range)/framerate)}_divfit.png')

            # Doubling time
            intervals = [0,75,100,150,175,225,250,325]
            bands = [i*framerate for i in intervals]

            doubling = np.log(2)/np.polyfit(x, np.log(y), 1)[0]
            if show:
                plt.figure(dpi=100)
                print(f'Doubling time is {doubling}')
                plt.plot(x, y, color=[0,0.4,0.6])
                # plt.vlines(bands,y[0],y[-1])
                plt.scatter(x, y, color=[0,0.4,0.6])
                plt.xlabel('Hours')
                plt.ylabel('Number of cells')
                plt.annotate(fr'Doubling time $\approx$ {doubling:2.1f} hrs', [doubling + 0.5,y[0]*3/2])
                plt.savefig(f'{f}_{int((start)/framerate)}_{int((start+ sample_range)/framerate)}_doubling.png')
            N_div_data.append([N0, m])

            start +=sample_range - overlap 
        div_dict.update({f:div_rate})
    elif apoptotsis:
        height =250
        data= pd.read_csv(f)
        plt.hist(resolution*np.sqrt(data.AreaShape_Area.values/np.pi), bins=100)
        plt.xlabel('R (um)')
        plt.ylabel('count')
        plt.vlines(resolution*np.sqrt(data.AreaShape_Area.values/np.pi).mean(),0,height, color = 'k')
        plt.annotate(f'R = {resolution*np.sqrt(data.AreaShape_Area.values/np.pi).mean():3.3} um',[resolution*np.sqrt(data.AreaShape_Area.values/np.pi).mean()+0.25,height], color = 'k')
        mean_r.append(np.sqrt(data.AreaShape_Area.values/np.pi).mean())
        plt.show()

