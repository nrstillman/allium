import allium 

import os 
import glob

import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
 
import networkx as nx
from scipy.spatial import distance_matrix

# TODO: 
#    - Seed sweep with single set of values
#    - Check saved adjacency matrix
# 
sweep = False
adj_from_saved = False  #off for now... need to work out what's gone wrong with adj matrix
param_bounds = [[0,0,0],[1, 10, 0.2]]
prior = allium.utils.init_prior(param_bounds)
N_samples = 1

if sweep:
    params = 'gnn/parameters.txt'
    for i,p in enumerate(prior.sample([N_samples])):
        print(i,', ' ,p,file=open(f'{params}', 'a'))
        folder = f'params_{i}'
        os.mkdir(f'gnn/{folder}')
        config = "include/config/simconfig_gnn.json"
        sim = allium.simulate.Sim(
            parameterFile = config, 
            folder=f'gnn/{folder}',
            save_prob=1,
            adjacency=True)
        a = sim.simulate(p)


folder = 'params_0'
files = glob.glob(f'gnn/{folder}/gnn_*.dat')
adjfiles = glob.glob(f'gnn/{folder}/adjMatrix_*.dat')
adjfiles.sort()
files.sort()

# creating a blank window
# for the animation
fig, ax = plt.subplots(dpi=200)
ax.clear()

df = pd.read_csv(files[0])
colors = [["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0] for n in range(len(df))]

if adj_from_saved:
    G = nx.from_numpy_array(np.loadtxt(adjfiles[0]))
else:
    dist_matrix = pd.DataFrame(distance_matrix(df[['x','y']].values,df[['x','y']]), index=df.index, columns=df.index)
    interaction_range = 1.2#params.cutoffZ
    adj_matrix = 1*(dist_matrix<interaction_range).values

    G = nx.from_numpy_array(adj_matrix)

pos = [(x,y) for x,y in df[['x','y']].values]
node_colors = [colors[n] for n in G.nodes]
# labels = df.flag

nx.draw(G,pos, node_size=100,
            node_color=node_colors, with_labels=True)
# plt.show()
# animation function
def animate(i):
    ax.clear()
    print(f'{i+1}',end='\r')
    df = pd.read_csv(files[i+1])
    
    if adj_from_saved:
        G = nx.from_numpy_array(np.loadtxt(adjfiles[i]))
    else:
        dist_matrix = pd.DataFrame(distance_matrix(df[['x','y']].values,df[['x','y']]), index=df.index, columns=df.index)
        interaction_range = 1.2#params.cutoffZ
        adj_matrix = 1*(dist_matrix<interaction_range).values

        G = nx.from_numpy_array(adj_matrix)

    pos = [(x,y) for x,y in df[['x','y']].values]
    node_colors = [colors[n] for n in G.nodes]
    labels = df.flag

    nx.draw(G, pos, node_size=df.radius*100,
                node_color=node_colors, with_labels=True)
 
# calling the animation function    
anim = animation.FuncAnimation(fig, animate,
                            frames = len(files)-1,
                            interval = 20)
 
# saves the animation in our desktop
anim.save(f'example_graph_adj_{adj_from_saved}.mp4', writer = 'ffmpeg', fps = 2)