import numpy as np
import matplotlib.pyplot as plt


def ApplyPeriodic2d(data,dr,replace=True):
	dr[:,0]-=data.param.Lx*np.round(dr[:,0]/data.param.Lx)
	dr[:,1]-=data.param.Ly*np.round(dr[:,1]/data.param.Ly)
	return dr

# By design, this is only meaningful on the whole data set, e.g. do not subtract for tracer particles only
def takeDriftFcn(data, Nvariable = True):
	data.drift=np.zeros((data.Nsnap,2))
	if Nvariable:
		print("Dynamics:: Variable N: Taking off the drift is meaningless. Doing nothing.")
	else:	
		for u in range(1,data.Nsnap):
			dr=ApplyPeriodic2d(data, data.rval[u,:,:]-data.rval[u-1,:,:])
			drift0=np.sum(dr,axis=0)/data.N
			data.drift[u,:]=data.drift[u-1,:]+drift0
	return data

# relative velocity distribution (and average velocity)
# component wise as well, assumes x and y directions only
def getVelDist(data,bins,bins2,usetype=[1],verbose=True):
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
		Nvariable = True

	if takeDrift:
		data = takeDriftFcn(data, Nvariable=Nvariable)
		
	for u in range(data.Nsnap):	
		if gettracers:
			tracers = data.gettypes(usetype,u) if gettracers else len(data.rval[u])
			Ntrack = len(tracers)

		smax=data.Nsnap-u
		# Note that by design, take drift needs to work on the whole data set
		if takeDrift:
			hmm=(data.drift[:smax,:]-data.drift[u:,:])
			takeoff=np.einsum('j,ik->ijk',np.ones((Ntrack,)),hmm)	
			dr=data.rval[:smax,tracers,:]-data.rval[u:,tracers,:]-takeoff[:,tracers,:]
		else:
			dr=data.rval[:smax,tracers,:]-data.rval[u:,tracers,:]

		msd[u]=np.sum(np.sum(np.sum(dr**2,axis=2),axis=1),axis=0)/(Ntrack*smax)

	data.hasMSD = True
	data.msd = msd

	xval=np.linspace(0,data.Nsnap*data.param.dt*data.param.output_time,num=data.Nsnap)
	if verbose:
		fig=plt.figure()
		plt.loglog(xval,msd,'r.-',lw=2)
		plt.loglog(xval,msd[1]/(1.0*xval[1])*xval,'-',lw=2,color=[0.5,0.5,0.5])
		plt.xlabel('time')
		plt.ylabel('MSD')
		plt.title('Mean square displacement')
		plt.show()

	return xval, msd, data

# Velocity autocorrelation function
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
	else:
		Nvariable = True

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
        

# Definition of the self-intermediate scattering function (Flenner + Szamel)
# 1/N <\sum_n exp(iq[r_n(t)-r_n(0)]>_t,n
def SelfIntermediate(data,qval,takeDrift,usetype=[1],verbose=True):
	# This is single particle, single q, shifted time step. Equivalent to the MSD, really
	SelfInt=np.empty((data.Nsnap,),dtype=complex)
	
	# Get tracers
	if data.Nvariable:
		if len(usetype) <1:
			print("Error: Cannot calculate MSD when number of particles is changing")
			sys.exit()
		else:
			gettracers = True
		
	for u in range(data.Nsnap):
		if gettracers:
			tracers = data.gettypes(usetype,u) 
			Ntrack = len(tracers)

		smax=data.Nsnap-u
		if takeDrift:
			hmm=(data.drift[:smax,:]-data.drift[u:,:])
			takeoff=np.einsum('j,ik->ijk',np.ones((Ntrack,)),hmm)

			SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(-data.rval[:smax,tracers,0]+data.rval[u:,tracers,0]+takeoff[:,tracers,0])+ \
											1.0j*qval[1]*(-data.rval[:smax,tracers,1]+data.rval[u:,tracers,1]+takeoff[:,tracers,1]) \
											)))
		else:
			print("Not implemented")
			# if data.geom.periodic:
			# 	SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(self.geom.ApplyPeriodicX(-self.rval[:smax,self.usethese,0]+self.rval[u:,self.usethese,0]))+1.0j*qval[1]*(self.geom.ApplyPeriodicY(-self.rval[:smax,self.usethese,1]+self.rval[u:,self.usethese,1]))+1.0j*qval[2]*(self.geom.ApplyPeriodicZ(-self.rval[:smax,self.usethese,2]+self.rval[u:,self.usethese,2]))),axis=1),axis=0)/(self.N*smax)
			# else:
			# 	SelfInt[u]=np.sum(np.sum(np.exp(1.0j*qval[0]*(-self.rval[:smax,self.usethese,0]+self.rval[u:,self.usethese,0])+1.0j*qval[1]*(-self.rval[:smax,self.usethese,1]+self.rval[u:,self.usethese,1])+1.0j*qval[2]*(-self.rval[:smax,self.usethese,2]+self.rval[u:,self.usethese,2])),axis=1),axis=0)/(Ntrack*smax)
                    
	# Looking at the absolute value of it here
	SelfInt2=(np.real(SelfInt)**2 + np.imag(SelfInt)**2)**0.5
	
	tval=np.linspace(0,data.Nsnap*data.param.dt*data.param.output_time,num=data.Nsnap)
	if verbose:
		qnorm=np.sqrt(qval[0]**2+qval[1]**2+qval[2]**2)
		fig=plt.figure()
		plt.semilogx(tval,SelfInt2,'.-r',lw=2)
		plt.xlabel('time')
		plt.ylabel('F_s(k,t)')
		plt.title('Self-intermediate, k = ' + str(qnorm))
		plt.show()
	return tval, SelfInt2

