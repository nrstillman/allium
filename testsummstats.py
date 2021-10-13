import allium 
import pickle
import numpy as np
import scipy

with open('output/v0_64_k_24_tau_5.p', 'rb') as f:
	d = pickle.load(f)

takeDrift = True

# remove any data post zap
d.truncateto(int(d.param.zaptime/d.param.output_time))

# # A - Velocity distributions and mean velocity
# # Bins are in normalised units (by mean velocity)
velbins=np.linspace(0,10,100)
velbins2=np.linspace(-10,10,100)
vav, vdist,vdist2 = allium.summstats.getVelDist(d, velbins,velbins2, usetype=[1],verbose=True)

# B - Autocorrelation Velocity Function
tval2, velauto, v2av = allium.summstats.getVelAuto(d, usetype=[1],verbose=True)

# C - Mean square displacement
tval, msd, d = allium.summstats.getMSD(d,takeDrift, usetype=[1],verbose=True)

# D - Self Intermediate Scattering Function
qval = 2*np.pi/1.0*np.array([1,0,0])
tval3, SelfInt = allium.summstats.SelfIntermediate(d, qval,True,usetype=[1],verbose=True)