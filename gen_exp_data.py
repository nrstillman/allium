# import matplotlib.pyplot as plt
# import pickle 
# import numpy as np
# #special numpy for neural networks
# import torch
# #neural network architecture for inference
# from sbi.inference import SNPE
# #only used to make pairplots
# from sbi import analysis

# import matplotlib.animation as animation
# import seaborn as sns
# import pandas as pd
import allium
import dill as pickle 
import time
import glob
files = glob.glob('experiment_data/*.p')

before_then = time.perf_counter()
for file in files:
	then = time.perf_counter()

	exp_data = '_'.join(file.split('_')[-2:])
	print(file)
	with open(f'experiment_data/neshika_exp_{exp_data}', 'rb') as f:
	    d = pickle.load(f)
	data = d[0]
	properties = d[1]

	output = allium.data.ExperimentData(data, properties)
	now = time.perf_counter()
	print(f'time taken for {exp_data} is {now-then}')

	with open(f'old_al_data_{exp_data}', 'wb') as f:
	    pickle.dump(output,f)

print(f'Total time taken is {exp_data} is {now-before_then}')
