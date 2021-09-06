import allium 
import pickle

data = allium.utils.load_exp()
time = np.linspace(0,95*5,94)
msd =allium.ss.calculate_msd(data,tracers=True, beginning = 0, end=95)
D = np.polyfit(np.log(time), np.log(msd[1:]), 1)[0]

with open('to_silke/sim_params.p', 'rb') as f:
        output = pickle.load(f)

with open('testposterior.p', 'rb') as f: 
        posterior = pickle.load(f)

exp_ss = [1.4418, 48.0926, 0.006875]
output_ss = output['ss']