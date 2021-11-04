import os 
import sys

files = os.listdir(folder)

folder = 'output/4params/'
file = '4paramsconfluent_v0_109_k_94_tau_4_ss.p'

for f in files:
   if f[6][-4:-3] == 's':
      with open(f"{folder}{file}", 'rb') as fin:
         d = pickle.load(fin)
      plt.plot(d['vav'][60:])

plt.show()
