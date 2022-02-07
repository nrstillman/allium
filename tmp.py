import allium 
import random
import glob
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
 
import networkx as nx
from scipy.spatial import distance_matrix


# TODO: 
#    - Parameter sweep with different values
#    - Seed sweep with single set of values

sweep = True
simulate = False

params = 
if simulate or sweep:
    config = "include/config/simconfig_gnn.json"
    sim = allium.simulate.Sim(parameterFile = config, save_prob=1)

    a = sim.simulate([0.01,1,10])

files = glob.glob('testfolder/*.dat')
files.sort()

# creating a blank window
# for the animation
fig, ax = plt.subplots(dpi=200)
ax.clear()

df = pd.read_csv(files[0])
colors = [["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0] for n in range(len(df))]

dist_matrix = pd.DataFrame(distance_matrix(df[['x','y']].values,df[['x','y']]), index=df.index, columns=df.index)

interaction_range = 1
adj_matrix = 1*(dist_matrix<interaction_range).values
rows, cols = np.where(adj_matrix == 1)
edges = zip(rows.tolist(), cols.tolist())

G = nx.Graph()
G.add_edges_from(edges)

pos = [(x,y) for x,y in df[['x','y']].values]
node_colors = [colors[n] for n in G.nodes]
labels = df.flag

nx.draw(G,pos, node_size=df.radius*100,
            node_color=node_colors, with_labels=True)

# nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="whitesmoke")

# animation function
def animate(i):
    ax.clear()
    print(f'{i+1}',end='\r')
    df = pd.read_csv(files[i+1])
    
    dist_matrix = pd.DataFrame(distance_matrix(df[['x','y']].values,df[['x','y']]), index=df.index, columns=df.index)
    
    interaction_range = 1.2
    adj_matrix = 1*(dist_matrix<interaction_range).values
    rows, cols = np.where(adj_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())

    G = nx.Graph()
    G.add_edges_from(edges)

    pos = [(x,y) for x,y in df[['x','y']].values]
    node_colors = [colors[n] for n in G.nodes]
    labels = df.flag

    nx.draw(G, pos, node_size=df.radius*100,
                node_color=node_colors, with_labels=True)
    # nx.draw_networkx_labels(G, pos, labels, font_size=22, font_color="whitesmoke")

 
# calling the animation function    
anim = animation.FuncAnimation(fig, animate,
                            frames = len(files)-1,
                            interval = 20)
 
# saves the animation in our desktop
anim.save('example_graph.mp4', writer = 'ffmpeg', fps = 2)