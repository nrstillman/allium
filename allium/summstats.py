import numpy as np
from scipy import optimize
from scipy import stats
import matplotlib.pyplot as plt


def calculate_summary_statistics(d, opts = ['A','B','C','D','E','F','G','H'],log=False,starttime=60,endtime=320,takeDrift=False, plot = False, usetypes = [1,2]):
    """
    Calculates summary statistics.

    """
    # 0 is new cells, 1 is tracer, 2 is original (check this)    
    # remove any data post zap
    d.truncateto(starttime, endtime)
    if takeDrift:
         d.takeDrift()

    ssdata = {}
    ssvect = []
    if 'A' in opts:
        # # # # # A - Velocity distributions and mean velocity
        velbins=np.linspace(0,10,100)
        velbins2=np.linspace(-10,10,100)
        vav, vdist,vdist2 = getVelDist(d, velbins,velbins2, usetype=usetypes,verbose=plot)
        
        vdist = vdist[1:]
        vdist2 = vdist2[vdist2 != max(vdist2)]

        ssdata['vav'] = vav
        ssdata['vdist'] = vdist
        ssdata['vdist2'] = vdist2
        if log: print('Finished calculating A. vel. dist & mean vel')
        ssvect.append(vav.mean()) 
        ssvect.append(stats.kurtosis(vdist,fisher=False))
        ssvect.append(vdist.mean())
        ssvect.append(vdist.var())
        ssvect.append(stats.kurtosis(vdist2,fisher=False))
        ssvect.append(vdist2.mean())
        ssvect.append(vdist2.var())
    if 'B' in opts:
        # # B - Autocorrelation Velocity Function
        tval2, velauto, v2av = getVelAuto(d, usetype=[1],verbose=plot)
        ssdata['tval2'] = tval2
        ssdata['velauto'] = velauto
        ssdata['v2av'] = v2av
        if log: print('Finished calculating B. autocorr vel fcn') 
        ssvect.append(tval2[velauto < 5e-1][0])
    if 'C' in opts:
        # C - Mean square displacement
        tval, msd, d = getMSD(d,takeDrift, usetype=[1],verbose=plot)
        ssdata['tval'] = tval
        ssdata['msd'] = msd
        if log: print('Finished calculating C. MSD') 
        ssvect.append(np.polyfit(np.log(tval[1:]), np.log(msd[1:]), 1)[0])
        ssvect.append(np.polyfit(np.log(tval[1:]), np.log(msd[1:]), 1)[1])
        ssvect.append(ssdata['msd'][-1])
        #v0, tau = optimize.curve_fit(lambda t, v0, tau:  2*v0*v0*tau*(t - tau*(1-np.exp(-t/tau))),xdata = tval[1:], ydata = msd[1:])[0]
        #ssvect.append(v0)
        #ssvect.append(tau)
    if 'D' in opts:     
        # # D - Self Intermediate Scattering Function
        qval = 2*np.pi/d.sigma*np.array([1,0])
        tval3, SelfInt2, SelfInt = SelfIntermediate(d, qval,takeDrift,usetype=[1],verbose=plot)
        ssdata['tval3'] = tval3
        ssdata['SelfInt2'] = SelfInt2
        ssdata['SelfInt'] = SelfInt
        step = 10
        qmax = np.pi/d.sigma #particle size in wavelength (upper limit)
        dx =  d.sigma*0.9
        xmax = d.param.Ly
        ssdata['dx'] = dx
        ssdata['xmax'] = xmax
        if log: print('Finished calculating D. self-intermediate scattering fcn') 
        if np.sum(SelfInt2 < 0.5) > 0:
            ssvect.append(tval3[SelfInt2 < 0.5][0])
        else:
            ssvect.append(tval3[-1])
    if 'E' in opts:
        # # E - real space velocity correlation function ('swirlyness')
        velcorrReal = np.zeros((150,))
        count = 0
        for u in range(0,endtime - starttime,step):
            # # # E - Real space velocity correlation function
            spacebins,velcorr = getVelcorrSingle(d, dx,xmax,whichframe=u,usetype=usetypes,verbose=plot)
            velcorrReal[:len(spacebins)] += velcorr  
            count+=1

        velcorrReal = velcorrReal[:len(spacebins)]
        velcorrReal/=count
        ssdata['velcorrReal'] = velcorrReal
        ssdata['spacebins'] = spacebins

        x = spacebins[(50<spacebins) & (spacebins < 300)]
        y = velcorrReal[(50<spacebins) & (spacebins< 300)]
        
        if log: print('Finished calculating E. vel. corr. fcn')
        if np.sum(y>0) > 0:
            ssvect.append(np.polyfit(np.log(x[y>0]), np.log(y[y>0]), 1)[0])
        else:
            ssvect.append(0)

    if 'F' in opts:
        # # F - Radial distribution function, g(r)
        rdist, gr = calcgr(d, verbose=plot)
        ssdata['rdist'] = rdist
        ssdata['gr'] = gr
        ssvect.append(rdist[np.where(gr == max(gr))][0])

    if 'G' in opts:
        # # G - Mean horizontal displacement
        if log: print('Finished calculating F. avg. horiz. disp. (from midway point)')
        ssvect.append(deltax(d))

    if 'H' in opts:
        # # H - Change in density
        if log: print('Finished calculating G. change in phi')
        ssvect.append(deltaphi(d))

    if 'I' in opts:
        # # I - average vector velocity
        ssvect.append(mean_vect_vel(d))

    if log: print('Finished calculating summary statistics')
    return ssvect, ssdata

def deltax(data,usetype=[1]):
    tracers_start = data.gettypes(usetype,0)
    tracers_end = data.gettypes(usetype,len(data.rval)-1)
    return np.mean(data.rval[-1][tracers_end,0] - data.rval[0][tracers_start,0])
    
def deltaphi(data):
    return (data.Nvals[-1] - data.Nvals[0])*(np.pi*data.param.R*data.param.R)/(data.param.Lx*data.param.Ly)        

def mean_vect_vel(data):
    return data.vval.mean(axis=1)

def ApplyPeriodic2d(data,dr):
    dr[:,0]-=data.param.Lx*np.round(dr[:,0]/data.param.Lx)
    dr[:,1]-=data.param.Ly*np.round(dr[:,1]/data.param.Ly)
    return dr

# relative velocity distribution (and average velocity)
# component wise as well, assumes x and y directions only
# use all:
def getVelDist(data,bins,bins2,usetype=[0,1],verbose=True):
    vav=np.zeros((data.Nsnap,))
    vdist=np.zeros((len(bins)-1,))
    vdist2=np.zeros((len(bins2)-1,))
    for u in range(data.Nsnap):
        # The particles we want to average over
        tracers = data.gettypes(usetype,u)
        # get all the magnitudes, and all the components
        vmagnitude=np.sqrt(data.vval[u,tracers,0]**2+data.vval[u,tracers,1]**2)
        vcomponents = data.vval[u,tracers,0:].flatten()
        # mean velocity magnitude
        vav[u]=np.mean(vmagnitude)
        # normalised magnitude histogram
        vdist0,dummy=np.histogram(vmagnitude/vav[u],bins,density=True)
        vdist+=vdist0
        # normalised component histogram
        vdist20,dummy=np.histogram(vcomponents/vav[u],bins2,density=True)
        vdist2+=vdist20
    vdist/=data.Nsnap
    vdist2/=data.Nsnap
    if verbose:
        fig=plt.figure()
        db=bins[1]-bins[0]
        plt.semilogy(bins[1:]-db/2,vdist,'r.-',lw=2)
        plt.xlabel('v/<v>')
        plt.ylabel('P(v/<v>)')
        plt.title('Scaled velocity magnitude distribution')
        plt.show()

        fig=plt.figure()
        db=bins2[1]-bins2[0]
        plt.semilogy(bins2[1:]-db/2,vdist2,'r.-',lw=2)
        plt.xlabel('v/<v>')
        plt.ylabel('P(v/<v>)')
        plt.title('Scaled velocity component distribution')
        plt.show()

        xval=np.linspace(0,data.Nsnap*data.param.dt*data.param.output_time,num=data.Nsnap)
        plt.figure()
        plt.plot(xval,vav,'r.-',lw=2)
        plt.xlabel('time')
        plt.ylabel('mean velocity')
        plt.show()

    return vav, vdist,vdist2        

# use tracers
def getMSD(data,takeDrift, usetype=[1],verbose=True):
    msd=np.empty((data.Nsnap,))
    # Get tracers
    if data.Nvariable:
        if len(usetype) <1:
            print("Error: Cannot calculate MSD when number of particles is changing")
            sys.exit()
        else:
            gettracers = True
            Nvariable= False
    else:
        Nvariable = False
        gettracers = True

    for u in range(data.Nsnap): 
        smax=data.Nsnap-u
        if gettracers:
            # get tracer idx for all times
            tracersidx = data.gettypes([1],range(data.Nsnap))
            # get number of tracers
            Ntrack = tracersidx[0].sum()
            # get rval for up to smax
            rt = data.rval[:smax,:][tracersidx[:smax,:]].reshape(smax, Ntrack,2)
            # get rval for u to end
            rtplus = data.rval[u:,:][tracersidx[u:,:]].reshape(smax, Ntrack, 2)

            dr  = rt - rtplus
            for n in range(smax):
                dr[n] = ApplyPeriodic2d(data, dr[n])

        msd[u]=np.sum(np.sum(np.sum(dr**2,axis=2),axis=1),axis=0)/(Ntrack*smax)

    data.hasMSD = True
    data.msd = msd
                    #careful with data.param here
    xval=np.linspace(0,data.Nsnap*data.param.framerate,num=data.Nsnap)
    if verbose:
        fig=plt.figure()
        plt.loglog(xval,msd,'r.-',lw=2)
        plt.loglog(xval,msd[1]/(1.0*xval[1])*xval,'-',lw=2,color=[0.5,0.5,0.5])
        plt.xlabel('time (hours)')
        plt.ylabel('MSD')
        plt.title('Mean square displacement')
        plt.show()

    return xval, msd, data

# Velocity autocorrelation function
# do use tracers
def getVelAuto(data,usetype=[1],verbose=True):
    velauto=np.empty((data.Nsnap,))
    v2av=np.empty((data.Nsnap,))
    
    # Get tracers
    if data.Nvariable:
        if len(usetype) <1:
            print("Error: Cannot calculate MSD when number of particles is changing")
            sys.exit()
        else:
            gettracers = True
            Nvariable= False

    # First compute normalised velocities. Note: normalised by mean velocity in the whole system at that time, not unit vectors!
    Ntrack = sum(data.gettypes(usetype,0))
    vnormed = np.zeros((data.Nsnap,Ntrack,2))

    for u in range(data.Nsnap):
        tracers = data.gettypes(usetype,u)
        v2av[u]=np.sum(np.sum((data.vval[u,tracers,:])**2,axis=1),axis=0)/(Ntrack)
        vnormed[u,:,:]=data.vval[u,tracers,:]/np.sqrt(v2av[u])
            
    for u in range(data.Nsnap):
        smax=data.Nsnap-u
        velauto[u]=np.sum(np.sum((vnormed[:smax,:,0]*vnormed[u:,:,0]+\
                                  vnormed[:smax,:,1]*vnormed[u:,:,1]),axis=1),axis=0)/(Ntrack*smax)
                            
    xval=np.linspace(0,data.Nsnap*data.param.dt*data.param.output_time,num=data.Nsnap)
    if verbose:
        fig=plt.figure()
        plt.loglog(xval,velauto,'r.-',lw=2)
        plt.xlabel('time')
        plt.ylabel('correlation')
        plt.title('Normalised Velocity autocorrelation function')
        plt.show()
    return xval, velauto, v2av        

def calcgr(data,  verbose = True, periodic=True, section = [150,550],resolution=2):
    
    def ApplyPeriodic2d(L,dr,):
        dr[0]-=L[0]*np.round(dr[0]/L[0])
        dr[1]-=L[1]*np.round(dr[1]/L[1])
        return dr

    def find_near(particle, xy, lower,upper, L):
        dr = np.sqrt((ApplyPeriodic2d(L,xy - particle)**2).sum(axis=1))
        return (lower < dr) & (dr < upper)

    max_distance = min(data.param.Lx,data.param.Ly)/4
    Nrings = int(max_distance/resolution)
    rdist = np.linspace(0,max_distance,Nrings)

    tbins = np.zeros(Nrings)

    area = np.zeros(Nrings)
    for j in range(Nrings):
        r1 = j * resolution
        area[j] = 2*np.pi*r1*resolution

    L = [l - 2*s for (l,s) in zip([data.param.Lx,data.param.Ly], section)]

    # loop through the frames and calculate g(r) 
    for t in range(data.Nsnap):
        # only consider a subsection of the data
        if len(section) > 0:
            ind = list(set(((section[0]<data.rval[t][:,1]) & 
                            (data.rval[t][:,1] <section[1]))*range(len(data.rval[t])))
               .intersection(((section[0]<data.rval[t][:,0]) &
                             (data.rval[t][:,0] <section[1]))*range(len(data.rval[t]))))

        #have to renormalise due to taking section
        x = data.rval[t][:,0][ind][1:] - section[0]
        y = data.rval[t][:,1][ind][1:] - section[1]
        xy = data.rval[t][ind][1:] - section

        for p in xy:
            for i,b in enumerate(tbins):
                r = i*resolution
                if (xy - p).shape[0] == 1:
                    continue
                else:
                    near =  find_near(p,xy, r-resolution, r, L)
                    tbins[i] += near.sum()

    gr = tbins[1:]/data.Nsnap/area[1:]
    # remove first incase equals nan   
    gr = gr[1:]
    # normalize such g(r) = 1 for r->inf 
    gr = gr/gr[-1]
    if verbose:
        fig=plt.figure()
        plt.plot(rdist, (tbins/data.Nsnap/area)/(tbins/data.Nsnap/area)[-1])
        # plt.xlim([0,max_distance])
        plt.xlabel('r')
        plt.ylabel('g(r)')
        plt.title('Static structure factor (2d)')
        plt.show()
    return rdist, gr


# Definition of the self-intermediate scattering function (Flenner + Szamel)
# 1/N <\sum_n exp(iq[r_n(t)-r_n(0)]>_t,n
def SelfIntermediate(data,qval,takeDrift,usetype=[1],verbose=True, periodic=True):
    # This is single particle, single q, shifted time step. Equivalent to the MSD, really
    SelfInt=np.empty((data.Nsnap,),dtype=complex)
    
    # Get tracers
    gettracers = True
    if data.Nvariable:
        if len(usetype) <1:
            print("Error: Cannot calculate MSD when number of particles is changing")
            sys.exit()
        else:
            gettracers = True
        
    for u in range(data.Nsnap):

        smax=data.Nsnap-u
        if gettracers:
            tracersidx = data.gettypes(usetype,range(data.Nsnap))
            Ntrack = tracersidx[0].sum()
            rt = data.rval[u:,:][tracersidx[u:,:]].reshape(smax, Ntrack, 2)
            rtplus = data.rval[:smax,:][tracersidx[:smax,:]].reshape(smax, Ntrack, 2)
            dr = rt- rtplus
            for n in range(smax):
                dr[n] = ApplyPeriodic2d(data, dr[n])

        if periodic:
            SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*dr[:,:,0]+ \
                                            1.0j*qval[1]*dr[:,:,1] \
                                        ),axis=1),axis=0)/(Ntrack*smax)         
        else:   
            SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(rt[:,:,0]-rtplus[:,:,0]) + \
                                            1.0j*qval[1]*(rt[:,:,1]-rtplus[:,:,1])\
                                            ),axis=1),axis=0)/(Ntrack*smax)                    

        
    # Looking at the absolute value of it here
    SelfInt2=(np.real(SelfInt)**2 + np.imag(SelfInt)**2)**0.5
    
    tval=np.linspace(0,data.Nsnap*data.param.dt*data.param.output_time,num=data.Nsnap)
    if verbose:
        qnorm=np.sqrt(qval[0]**2+qval[1]**2)
        fig=plt.figure()
        plt.semilogx(tval,SelfInt2,'.-r',lw=2)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate, k = ' + str(qnorm))
        plt.ylim([0,1])
        plt.show()

        fig=plt.figure()
        plt.plot(tval,SelfInt2,'.-r',lw=2)
        plt.xlabel('time')
        plt.ylabel('F_s(k,t)')
        plt.title('Self-intermediate, k = ' + str(qnorm))
        plt.ylim([0,1])
        plt.show()
    return tval, SelfInt2, SelfInt

####################### Fourier space and real space equal time correlation functions ##################################

# Generate 2d points for radially averaged Fourier transform computations
def makeQrad(dq,qmax,nq):
    nq2=int(2**0.5*nq)
    qmax2=2**0.5*qmax
    qx=np.linspace(0,qmax,nq)
    qy=np.linspace(0,qmax,nq)
    qrad=np.linspace(0,qmax2,nq2)
    # do this silly counting once and for all
    binval=np.empty((nq,nq))
    for kx in range(nq):
        for ky in range(nq):
            qval=np.sqrt(qx[kx]**2+qy[ky]**2)
            binval[kx,ky]=round(qval/dq)
    ptsx=[]
    ptsy=[]
    # do the indexing arrays
    for l in range(nq2):
        pts0x=[]
        pts0y=[]
        for kx in range(nq):
            hmm=np.nonzero(binval[kx,:]==l)[0]
            for v in range(len(hmm)):
                pts0y.append(hmm[v])
                pts0x.append(kx)
        ptsx.append(pts0x)
        ptsy.append(pts0y)
    return qx, qy, qrad, ptsx, ptsy

# Static structure factor
# Which is implicitly in 2D!!
# FourierTrans(g(R)) = S(q)
def FourierTrans(data,qmax=0.3,whichframe=1,usetype=[0,1],verbose=True):
    
    L = data.param.Lx

    # Note to self: only low q values will be interesting in any case. 
    # The stepping is in multiples of the inverse box size. Assuming a square box.
    print("Fourier transforming positions")
    dq=2*np.pi/L
    nq=int(qmax/dq)
    print("Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps.")
    qx, qy, qrad, ptsx, ptsy=makeQrad(dq,qmax,nq)
    #print " After Qrad" 
    fourierval=np.zeros((nq,nq),dtype=complex)
    
    #index relevant particles (by default we use all of them)
    useparts = data.gettypes(usetype,whichframe)
    N = sum(useparts)
    for kx in range(nq):
        for ky in range(nq):
            # And, alas, no FFT since we are most definitely off grid. And averaging is going to kill everything.
            fourierval[kx,ky]=np.sum(np.exp(1j*(qx[kx]*data.rval[whichframe,useparts,0]+qy[ky]*data.rval[whichframe,useparts,1])))/N
    plotval=N*(np.real(fourierval)**2+np.imag(fourierval)**2)
    
    # Produce a radial averaging to see if anything interesting happens
    nq2=int(2**0.5*nq)
    valrad=np.zeros((nq2,))
    for l in range(nq2):
        valrad[l]=np.mean(plotval[ptsx[l],ptsy[l]])#, axis=0)
    
    if verbose:
        plt.figure()
        plt.pcolor(qx,qy,plotval, vmin=0, vmax=3,shading='auto' )
        plt.colorbar()
        plt.title('Static structure factor (2d)')
        
        plt.figure()
        plt.plot(qrad,valrad,'.-r',lw=2)
        plt.xlabel('q')
        plt.ylabel('S(q)')
        plt.title('Static structure factor (radial)')
        
    return qrad,valrad
  
#use all
def FourierTransVel(data,qmax=0.3,whichframe=1,usetype='all',verbose=True):
    
    L = data.param.Lx

    # Note to self: only low q values will be interesting in any case. 
    # The stepping is in multiples of the inverse box size. Assuming a square box.
    print("Fourier transforming velocities")
    dq=2*np.pi/L
    nq=int(qmax/dq)
    print("Stepping Fourier transform with step " + str(dq)+ ", resulting in " + str(nq)+ " steps.")
    qx, qy, qrad, ptsx, ptsy=makeQrad(dq,qmax,nq)
    #print " After Qrad" 
    fourierval=np.zeros((nq,nq,2),dtype=complex)

    #index relevant particles (by default we use all of them)
    useparts = data.gettypes(usetype,whichframe)
    N = sum(useparts)
    for kx in range(nq):
        for ky in range(nq):
            fourierval[kx,ky,0]=np.sum(np.exp(1j*(qx[kx]*data.rval[whichframe,useparts,0]+qy[ky]*data.rval[whichframe,useparts,1]))*data.vval[whichframe,useparts,0])/N
            fourierval[kx,ky,1]=np.sum(np.exp(1j*(qx[kx]*data.rval[whichframe,useparts,0]+qy[ky]*data.rval[whichframe,useparts,1]))*data.vval[whichframe,useparts,1])/N 
    
    # Sq = \vec{v_q}.\vec{v_-q}, assuming real and symmetric
    # = \vec{v_q}.\vec{v_q*} = v
    Sq=np.real(fourierval[:,:,0])**2+np.imag(fourierval[:,:,0])**2+np.real(fourierval[:,:,1])**2+np.imag(fourierval[:,:,1])**2
    Sq=N*Sq
    # Produce a radial averaging to see if anything interesting happens
    nq2=int(2**0.5*nq)
    Sqrad=np.zeros((nq2,))
    for l in range(nq2):
        Sqrad[l]=np.mean(Sq[ptsx[l],ptsy[l]])
    
    plotval_x=np.sqrt(np.real(fourierval[:,:,0])**2+np.imag(fourierval[:,:,0])**2)
    plotval_y=np.sqrt(np.real(fourierval[:,:,1])**2+np.imag(fourierval[:,:,1])**2)
    # Produce a radial averaging to see if anything interesting happens
    valrad=np.zeros((nq2,2))
    for l in range(nq2):
        valrad[l,0]=np.mean(plotval_x[ptsx[l],ptsy[l]])
        valrad[l,1]=np.mean(plotval_y[ptsx[l],ptsy[l]])
    if verbose:
        plt.figure()
        plt.plot(qrad,Sqrad,'.-r',lw=2)
        plt.xlabel('q')
        plt.ylabel('correlation')
        plt.title('Fourier space velocity correlation')
    return qrad,valrad,Sqrad

# Real space velocity correlation function
# Note that this can work in higher dimensions. Uses geodesic distance, i.e. on the sphere if necessary

def getVelcorrSingle(data,dx,xmax,whichframe=1,usetype='all',verbose=True):
    # start with the isotropic one - since there is no obvious polar region
    # and n is not the relevant variable, and v varies too much
    # print("Velocity correlation function for frame " + str(whichframe))
    npts=int(round(xmax/dx))
    bins=np.linspace(0,xmax,npts)
    velcorr=np.zeros((npts,))
    velcount=np.zeros((npts,))
    #index relevant particles (by default we use all of them)
    useparts = data.gettypes(usetype,whichframe)
    N = sum(useparts)
    velav=np.mean(data.vval[whichframe,useparts,:],axis=0)

    for k in range(N):
        
        vdot=np.dot(data.vval[whichframe,useparts,:],data.vval[whichframe,useparts,:][k])
        
        ##Discretise spatially wrt particle distance 
        #ApplyPeriodicBC and take norm
        dr = ApplyPeriodic2d(data, data.rval[whichframe,useparts,:]- data.rval[whichframe,useparts,:][k])
        dr = np.linalg.norm(dr,axis=1)
        #calculate number of bins
        drbin=(np.round(dr/dx)).astype(int)
        #binning velocity correlations based on interparticle distance
        for l in range(npts):
            pts=np.nonzero(drbin==l)[0]
            velcorr[l]+=vdot[pts].sum()
            velcount[l]+=len(pts)
    
    isdata=[index for index, value in enumerate(velcount) if value>0]
    #connected correlation fcn
    velcorr[isdata]=velcorr[isdata]/velcount[isdata] #- np.dot(velav,velav)
    if verbose:
        fig=plt.figure()
        isdata=[index for index, value in enumerate(velcount) if value>0]
        plt.loglog(bins[isdata],velcorr[isdata],'.-r',lw=2)
        #plt.show()
        plt.xlabel("r-r'")
        plt.ylabel('Correlation')
        plt.title('Spatial velocity correlation')
    return bins,velcorr


