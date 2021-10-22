import allium 
import pickle
import numpy as np

from scipy import stats
import matplotlib.pyplot as plt

#sqrt(k)*tau/zeta
# very nearly flocking - correlates w system size

with open('v0_86_k_109_tau_9.p', 'rb') as f:
# with open('output/v0_126_k_71_tau_8.p', 'rb') as f:
	d = pickle.load(f)

# use tracers
# drift > print out how much is taken off (shouldn't be more than 2*sigma)
# see if removing it makes any difference

# check all normalization factors (check what they should be)
# should be # of frames & # of tracers

takeDrift = False #< check this
plot = True
starttime = 60
endtime = 320
# 0 is new cells, 1 is tracer, 2 is original (check this)
usetypes = [0,1,2]
end = int(d.param.zaptime/d.param.output_time) #320
# remove any data post zap
d.truncateto(starttime, endtime)

# # # # A - Velocity distributions and mean velocity
# # # # Bins are in normalised units (by mean velocity)
velbins=np.linspace(0,10,100)
velbins2=np.linspace(-10,10,100)
vav, vdist,vdist2 = allium.summstats.getVelDist(d, velbins,velbins2, usetype=usetypes,verbose=plot)

# # B - Autocorrelation Velocity Function
tval2, velauto, v2av = allium.summstats.getVelAuto(d, usetype=[1],verbose=plot)

# C - Mean square displacement
tval, msd, d = allium.summstats.getMSD(d,takeDrift, usetype=[1],verbose=plot)

# D - Self Intermediate Scattering Function
# offline until I check w Silke
qval = 2*np.pi/d.sigma*np.array([1,0])
tval3, SelfInt2, SelfInt = allium.summstats.SelfIntermediate(d, qval,takeDrift,usetype=[1],verbose=plot)

step = 10
# #had to increase qmax to be greater than 
qmax = np.pi/d.sigma #particle size in wavelength (upper limit)
# qmax *= 0.5 #interested in lower q values

dq=2*np.pi/d.param.Ly #use smaller (for computation sake)
nq=int(qmax/dq)

qs = allium.summstats.makeQrad(dq, qmax, nq)# can determine size from makeQrad<but which value?
size_of_vars = len(qs[2])
# # F - static structure factor
structurefact = np.zeros((500,))
# # G - velocity correlation function in Fourier space
velcorrFourier = np.zeros((500,))
# # H - real space velocity correlation function ('swirlyness')
velcorrReal = np.zeros((100,))
count = 0
plot = False
# for u in range(0,endtime - starttime,step):
#     print(u)
#     # if u == endtime - starttime:
#     #         plot = True
#     # F - Static structure factor, i.e. the Fourier transform of g(r) = S(q)
#     # if don't use all then holes in structure
#     # qrad,valrad = FourierTrans(self,qmax=0.3,whichframe=1,usetype='all',L="default",verbose=True):

#     qrad,valrad0 = allium.summstats.FourierTrans(d,qmax=qmax,whichframe=u,usetype=usetypes,verbose=plot)
#     structurefact[:len(qrad)] += valrad0

#     # G - Fourier space velocity correlation function
#     # better to use all to increase sample size
#     #qrad,valrad,Sqrad=FourierTransVel(self,qmax=0.3,whichframe=1,usetype='all',L="default",verbose=True)
#     qrad2,valrad0,Sqrad=allium.summstats.FourierTransVel(d,qmax=qmax,whichframe=u,usetype=usetypes,verbose=plot)
#     velcorrFourier[:len(qrad2)] += Sqrad
    
#     # H - Real space velocity correlation function
#     # better to use all to increase sample size
#     # requires periodic BC (difficult in applying to exp)
#     # bins,velcorr = getVelcorrSingle(self,dx,xmax,whichframe=1,usetype='all',verbose=True):
#     # spacing < 1 cell radius, out to 50 cell radii
#     spacebins,velcorr = allium.summstats.getVelcorrSingle(d, d.sigma/2,50*d.sigma,whichframe=u,usetype=usetypes,verbose=plot)
#     velcorrReal[:len(spacebins)] += velcorr
#     print(velcorrReal[1])
    
#     count+=1

# structurefact/=count
# velcorrFourier/=count
# velcorrReal/=count

# plt.figure()
# plt.semilogy(qrad,structurefact[:123],'.-r',lw=2)
# plt.xlabel('q')
# plt.ylabel('correlation')
# plt.title('Structure Factor')
# #complex
# plt.figure()
# plt.loglog(qrad2,velcorrFourier[:123],'.-r',lw=2)
# plt.xlabel('q')
# plt.ylabel('correlation')
# plt.title('Fourier space velocity correlation')

# # #exponential
# plt.figure()
# plt.semilogy(spacebins,velcorrReal[:100],'.-r',lw=2)
# plt.xlabel('r')
# plt.ylabel('correlation')
# plt.title('Spatial velocity correlation')

# plt.figure()
# plt.loglog(spacebins,velcorrReal[:100],'.-r',lw=2)
# plt.xlabel('r')
# plt.ylabel('correlation')
# plt.title('Spatial velocity correlation')

# # Self intermediate w single time scale, \ (PIV)
# #showing inflection point representing liquid phase (majority of particles move 1R from starting point)
# # tau_alpha = tval3[SelfInt2 < 0.5][0]

# # Autocorrelation @ 0.1\
# #any correlation between particles approaches 0 (note, 1e-1 is due to fast drop off)
# # persistance_time = tval2[velauto < 1e-1][0]

ss = [vav.mean(),
      stats.kurtosis(vdist,fisher=False),vdist.mean(),vdist.var(),\
      stats.kurtosis(vdist2,fisher=False),vdist2.mean(),vdist2.var(),\
      np.polyfit(np.log(tval[1:]), np.log(msd[1:]), 1)[0], \
      # tval3[SelfInt2 < 0.5][0],\
        # tval2[velauto < 1e-2][0],\
      ]

print(f'{ss}')