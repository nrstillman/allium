import allium 

import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
 
import networkx as nx
from scipy.spatial import distance_matrix

simulate = False
if simulate:
    config = "include/config/simconfig_gnn.json"
    sim = allium.simulate.Sim(parameterFile = config, save_prob=1)

    a = sim.simulate([0.01,1,10])

files = glob.glob('testfolder/*.dat')
files.sort()

# creating a blank window
# for the animation
fig, ax = plt.subplots()
ax.clear()

df = pd.read_csv(files[0])

dist_matrix = pd.DataFrame(distance_matrix(df[['x','y']].values,df[['x','y']]), index=df.index, columns=df.index)

interaction_range = 1
adj_matrix = 1*(dist_matrix<interaction_range).values

rows, cols = np.where(adj_matrix == 1)
edges = zip(rows.tolist(), cols.tolist())
gr = nx.Graph()
gr.add_edges_from(edges)
nx.draw(gr,[(x,y) for x,y in df[['x','y']].values], node_size=75)

# animation function
def animate(i):
    ax.clear()
    print(f'{i+1}',end='\r')
    df = pd.read_csv(files[i+1])
    
    dist_matrix = pd.DataFrame(distance_matrix(df[['x','y']].values,df[['x','y']]), index=df.index, columns=df.index)
    
    interaction_range = 1.3
    adj_matrix = 1*(dist_matrix<interaction_range).values

    rows, cols = np.where(adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.Graph()
    gr.add_edges_from(edges)
    nx.draw(gr,[(x,y) for x,y in df[['x','y']].values], node_size=50)

 
# calling the animation function    
anim = animation.FuncAnimation(fig, animate,
                            frames = len(files)-1,
                            interval = 20)
 
# saves the animation in our desktop
anim.save('example_graph.mp4', writer = 'ffmpeg', fps = 2)