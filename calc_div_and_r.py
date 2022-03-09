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

files =  glob.glob('segments/clean/*.csv')#glob.glob('./segmenting/fast/*.csv')

division = True
apoptotsis = False
div_rate = []
mean_r = []
resolution = 0.8
for f in files:
    print(f)
    # if (f.split('/')[3][0] == 'c') and (division):
    if division:
        data= pd.read_csv(f)
        print(f'N0 = {len(data[data.t == 1])},Nmax = {len(data[data.t == 149])}')
        ss = summary_statistics(data)
        framerate = 10/60
        x =np.unique(data['t'].values)*framerate # [i*30/60 for i in range(ss['Tmax'])]
        y = data.groupby(['t']).count().values[:,0]#ss['Ncells']
        
        #Slope calculation
        print(f"Slope = {np.polyfit(x, np.log(y), 1)[0]:.2}" )
        m = np.polyfit(x, np.log(y), 1)[0]      
        plt.figure(dpi=100)
        plt.plot(x, np.poly1d(np.polyfit(x, np.log(y), 1))(x), 'r')
        plt.scatter(x,np.log(y))
        # plt.title(f)
        plt.xlabel('Hours')
        plt.ylabel('log(Number of cells)')
        plt.annotate(m, [15,np.log(y[36])], rotation=20)
        plt.savefig(f'{f}divfit.png')
        # Doubling time
        doubling = np.log(2)/np.polyfit(x, np.log(y), 1)[0]
        plt.figure(dpi=100)
        print(f'Doubling time is {doubling}')
        plt.plot(x, y, color=[0,0.4,0.6])
        plt.vlines(doubling,y[0],2*y[0])
        plt.scatter(x, y, color=[0,0.4,0.6])
        plt.xlabel('Hours')
        plt.ylabel('Number of cells')
        plt.annotate(fr'Doubling time $\approx$ {doubling:2.1f} hrs', [doubling + 0.5,y[0]*3/2])
        plt.savefig(f'{f}_doubling.png')
        div_rate.append(m)
        print(div_rate)
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

plt.show()
