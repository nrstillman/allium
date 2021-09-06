import numpy as np
import pandas as pd
import glob
import json

def distance(a, b, Lx, Ly):
    dx = abs(a[0] - b[0])
    x = min(dx, abs(Lx - dx))

    dy = abs(a[1] - b[1])
    y = min(dy, abs(Ly - dy))

    return np.sqrt(x**2 + y**2)

def exp_ss(data, exp_length = 95):
    track_length = []
    #flags here are those tracks that span the entirety of the dataset
    flags = []
    for f in data.flag.unique():
        print('\r'+str(f),end='')
        track_length.append(sum(data.flag == f))
        if sum(data.flag == f) == exp_length:
            flags.append(f)
    msd = calculate_msd(data,flags=flags)
    return msd

def average_horizontal_displacement(data, tracers=True, beg = 0, end = 319):
    # prefer to use tracers particles
    data = data[beg:end]
    if tracers:
        t0_flag =  data[0][data[0][:,-1] == 1][:,0]
    # only take original cells (note - some of these may have died? see condition below)
    else:
        t0_flag =  data[0][:,0]

    #Error check for missing flags ... TODO
    # tf_flag  = flags = data[-1][data[-1][:,-1] == 1][:,0]    
    # no_death = sum([any(tf_flag.values == f) for f in t0_flag])
    # if no_death < len(t0_flag):
    #     return print('missing flags - check input')
        
    indices = []
    dx = []
    for f in t0_flag:
        tf=  data[-1]
        t0=  data[0]
        dx.append(np.abs(tf[tf[:,0] == f][:,1]  - t0[t0[:,0] == f][:,1]))
    return np.asarray(dx).mean() 

def change_in_phi(output):
    return (len(output['data'][-1]) - len(output['data'][0]))*output['params']['R']/(output['params']['Lx']*output['params']['Ly'])

def calculate_msd(data, tracers=True, flags=None, beg = 0, end=319):
    if flags is None:
        if tracers:
            flags = data[beg][data[beg][:,-1] == 1][:,0]
        else:
            flags = data[beg][:,0]

    msd = []
    for f in flags:
        r = []
        #there is a better way to do this...
        if type(data) is list:
            calculate = True
            for i, d in enumerate(data[beg:end]):
                xy = d[d[:,0] == f][:,1:3]
                if not bool(len(xy)):
                    print(f'flag {f} not found in frame {i}')
                    calculate = False
                    break
                else:
                    r.append(xy[0])
            if calculate:
                msd.append(msd_fft(np.asarray(r)))
        else:
            xy = data[data.flag == f][['x','y']].values.astype(float)
            msd.append(msd_fft(xy))

    return np.array(msd).mean(axis=0) 


def calculate_gr(data, tracers=False, flags=None, Lx=1600, Ly = 800):
    #this does not work...
    r_cutoff = min(Lx, Ly) / 2.0
    resolution = 1000
    dr = r_cutoff/resolution
    radii = np.linspace(0.0, r_cutoff, resolution)
    area = np.zeros(resolution)
    gr = np.zeros(resolution)
    if flags is None:
        if tracers:
            flags = data[0][data[0].type =='2'].flag.values
        else:
            flags = data[0].flag.values

    #there is a better way to do this...
    if type(data) is list:
        n = 1
        for d in data:
            area_per_p = Lx*Ly/(len(d))
            xy = np.array(d[['x','y']].values.astype(float))
            for i, p1 in enumerate(xy):
                for j in range(resolution):
                    area[j] += 2*np.pi*(j+1)*dr
                for p2 in xy[i:]:
                    dist = distance(p1, p2, Lx, Ly)
                    index = int(dist / dr)
                    if 0 < index < resolution:
                        #pairs so add 2
                        gr[index] += 2.
            n +=1
        #normalize
        for i, value in enumerate(gr):
            gr[i] = value * area_per_p / area[i]

    else:
        print('doesnt work for sim data yet')

    return gr

def msd_straightforward(r):
    shifts = np.arange(len(r))
    msds = np.zeros(shifts.size)    

    for i, shift in enumerate(shifts):
        diffs = r[:-shift if shift else None] - r[shift:]
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()

    return msds

def autocorrFFT(x):
  N=len(x)
  F = np.fft.fft(x, n=2*N)  #2*N because of zero-padding
  PSD = F * F.conjugate()
  res = np.fft.ifft(PSD)
  res= (res[:N]).real   #now we have the autocorrelation in convention B
  n=N*np.ones(N)-np.arange(0,N) #divide res(m) by (N-m)
  return res/n #this is the autocorrelation in convention A

def msd_fft(r):
  N=len(r)
  D=np.square(r).sum(axis=1) 
  D=np.append(D,0) 
  S2=sum([autocorrFFT(r[:, i]) for i in range(r.shape[1])])
  Q=2*D.sum()
  S1=np.zeros(N)
  for m in range(N):
      Q=Q-D[m-1]-D[N-m]
      S1[m]=Q/(N-m)
  return S1-2*S2
