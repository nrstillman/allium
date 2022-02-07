import numpy as np
class Parameters(object):
	def __init__(self, p):
		for key, values in p.items():
			setattr(self, key, values)

# global param #required for pickling
class param:
	def __init__(self):
		self.Lx = 800
		self.Ly = 800
		self.R = 8
		self.framerate = 0.083
		self.dt = 0.001
		self.output_time = self.framerate/self.dt

class ExperimentData:

	def gettypes(self, readtypes, frames):
		return np.isin(self.ptype[frames],readtypes)
		
	def truncateto(self,start, endtime):
		self.Nsnap = endtime - start
		self.Nvariable = False
		self.flag =  self.flag[start:endtime]
		self.rval = self.rval[start:endtime]
		self.vval = self.vval[start:endtime]
		# self.radius = self.radius[start:endtime]
		self.ptype = self.ptype[start:endtime]

	def __init__(self,data,properties):

		self.param = param()
		self.sigma = 8
		self.umpp = 0.8
		self.framerate = 0.083 #hours

		self.Nvals = [data[data[:,1] == n].shape[0] for n in range(properties['t'][-1])]
		maxNcells = max(self.Nvals)

		self.rval = np.zeros((properties['t'][-1]-1,maxNcells,2))
		self.flag = np.zeros((properties['t'][-1]-1,maxNcells))
		self.ptype = np.zeros((properties['t'][-1]-1,maxNcells))
		d = []
		tracers = []

		for n in range(1,properties['parent'][-1]):
			if len(data[data[:,0] == n]) == (properties['t'][-1]):
				tracers.append(n)

		for n in range(1,properties['t'][-1]):
			Ncells = len(data[data[:,1] == n])
			self.rval[n-1,:Ncells,:] = data[data[:,1] == n][:,2:]*self.umpp
			self.flag[n-1,:Ncells] = data[data[:,1] == n][:,0]
			self.ptype[n-1,:] = np.asarray([int(n in tracers) for n in self.flag[n-1]])

		# self.vval = np.zeros((properties['t'][-1]-2,maxNcells,2))
		# for n in range(0, properties['t'][-1]-2):
		# 	self.vval[n,self.ptype[n] == 1,:]  = (self.rval[n+1][self.ptype[n+1] == 1] - self.rval[n][self.ptype[n] == 1])
		# 	self.vval[n,self.ptype[n] == 1,:] *= (60/self.framerate) #converts from um/min to um/hour 
		# # TODO: new_from_here
		self.max_N_flag = self.flag[-1][-1]
		#loop through and only copy rvalues for those trajectories longer than 5 steps
		flags = []
		for flag in range(int(self.max_N_flag)):
		    time = np.linspace(1,self.flag.shape[0],self.flag.shape[0])*(self.flag == flag).sum(axis=1)
		    time = np.asarray(time)
		    
		    if sum(time!=0) > 5:
		        print(flag, end='\r')
		        flags.append(flag)
		print('Quick first stage finished')
		rvalues = []
		timevalues = []
		velvalues = []
		#generate flag specific rvalues,timevalues and velvalues
	
		for flag in flags[1:]:
		    print(flag, end='\r')
		    # get all rvalues related to a flag and turn into a numpy array
		    rval = np.concatenate([r[self.flag[t] == flag] for t,r in enumerate(self.rval)],axis=0)
		    # get all time values related to a flag and +1 to avoid removing 0th time later on
		    time = np.asarray([(t+1)*sum(self.flag[t] == flag) for t,r in enumerate(self.rval)])
		    
		    # append everything but the first which is removed as there is no veloc data there
		    rvalues.append(rval[1:])
		    # note, we don't take any of the 0 time values that end up padding based on bool check above
		    timevalues.append(time[time !=0][1:])
		    velvalues.append(np.diff(rval, axis=0))

		# ^data now ordered by flag number
		print('Slow second stage finished (should be quick from now).')
		# length of all flags
		self.flag_lifespan = np.asarray([len(r) for r in rvalues])

		self.max_T = self.flag_lifespan.max()
		r_new = []
		flag_new = []
		v_new = []

		#Reorder to become time
		for t in range(2,self.max_T):
		    print(t,end='\r')
		    flagtmp = []
		    rtmp =[]
		    vtmp = []
		    for i, flag in enumerate(flags[1:]):
		        if bool(sum(timevalues[i]==t)):
		            flagtmp.append(flag)
		            rtmp.append(rvalues[i][timevalues[i]==t])            
		            vtmp.append(velvalues[i][timevalues[i]==t])            
		    flag_new.append(np.asarray(flagtmp))
		    r_new.append(np.concatenate(rtmp,axis=0))
		    v_new.append(np.concatenate(vtmp,axis=0))

		# number of flags per timestep
		self.flags_per_timestep = np.asarray([len(flag) for flag in flag_new])
		self.maxN = self.flags_per_timestep.max()

		self.rval = np.zeros((self.max_T-2, self.flags_per_timestep.max(),2))
		self.vval = np.zeros((self.max_T-2, self.flags_per_timestep.max(),2))
		self.flag = np.zeros((self.max_T-2, self.flags_per_timestep.max()))
		#Turn back into arrayss
		for t in range(self.max_T-2):
		    self.rval[t,:len(r_new[t]),:] = r_new[t]
		    self.vval[t,:len(v_new[t]),:] = v_new[t]
		    self.flag[t,:len(flag_new[t])] =  flag_new[t]


class SimData:
	def checkTypes(readtypes,data):
		#check which particles to load 
		if len(readtypes) > 0:
			usetypes = np.isin(data[:,-1],readtypes)
		else:
			usetypes = [True]*len(data)
		return usetypes

	# Data object for summary statistics
	def __init__(self,**kwargs):
		# check for debugging
		try:
			self.debug = kwargs['debug']
			if self.debug:
				print('kwargs: ', kwargs)
		except:
			self.debug = False
		# check for specific loadtimes
		try:	
			self.start = kwargs["loadtimes"][0]
			self.end = kwargs["loadtimes"][1]
			self.multiopt = True
		except:
			self.multiopt = False
		# check for specific types
		try:
			self.readtypes = kwargs["readtypes"]
		except:
			self.readtypes = []
		# load parameters
		try:	
			self.param = Parameters(kwargs['params'])
		except:
			print('Error! Parameters must be a dictionary')
			return 1
		# load multiple simulation snapshots
		if self.multiopt:
			self.Nsnap = self.end - self.start
			#get maximum number of particles
			self.N = sum(SimData.checkTypes(self.readtypes, kwargs['data'][0]))
			self.Nvals = []
			self.Nvariable =  False
			for t in range(self.start,self.end):
				self.Nvals.append(sum(SimData.checkTypes(self.readtypes, kwargs['data'][t])))
				if self.Nvals[t] > self.N:
					self.N = self.Nvals[t] 
					self.Nvariable = True

			self.flag=np.zeros((self.Nsnap,self.N))
			self.rval=np.zeros((self.Nsnap,self.N,2))
			self.vval=np.zeros((self.Nsnap,self.N,2))
			self.theta =np.zeros((self.Nsnap,self.N))
			self.nval=np.zeros((self.Nsnap,self.N,2))
			self.radius=np.zeros((self.Nsnap,self.N))
			self.ptype=np.zeros((self.Nsnap,self.N))
			self.sigma = 0.

			for t in range(self.start,self.end):
				# only get particles we're interestsed in
				usetypes = SimData.checkTypes(self.readtypes, kwargs['data'][t])
				
				idx = range(sum(usetypes))
				#check whether data is old or new style
				if kwargs['data'][t].shape[1] > 4:
					#new output includes v,theta,radius
					self.flag[t,idx] =  kwargs['data'][t][usetypes,0]
					self.rval[t,idx,:] = kwargs['data'][t][usetypes,1:3]
					self.vval[t,idx,:] = kwargs['data'][t][usetypes,3:5]
					self.theta[t,idx] = kwargs['data'][t][usetypes,5]
					self.nval[t,idx,:] = np.array([np.cos(kwargs['data'][t][usetypes,5]), np.sin(kwargs['data'][t][usetypes,5])]).T
					self.radius[t,idx] = kwargs['data'][t][usetypes,6]
					self.ptype[t,idx] = kwargs['data'][t][usetypes,7]
					sigma = np.mean(kwargs['data'][t][usetypes,6])
					if sigma>self.sigma:
						self.sigma = sigma
				else:
					#old output only contains flag, r and type
					self.flag[t,idx] =  kwargs['data'][t][usetypes,0]
					self.rval[t,idx,:] = kwargs['data'][t][usetypes,1:3]
					self.ptype[t,idx] = kwargs['data'][t][usetypes, 3]

		# or a single snapshot
		else:
			# only get particles we're interestsed in
			usetypes = SimData.checkTypes(self.readtypes, kwargs['data'])
			self.Ntrack = sum(usetypes)
			#check whether data is old or new style
			if kwargs['data'].shape[1] > 4:
				#new output includes v,theta,radius
				self.flag =  kwargs['data'][usetypes,0]
				self.rval = kwargs['data'][usetypes,1:3]
				self.vval = kwargs['data'][usetypes,3:5]
				self.theta = kwargs['data'][usetypes,5]
				self.nval = np.array([np.cos(self.theta), np.sin(self.theta)]).T
				self.radius = kwargs['data'][usetypes,6]
				self.ptype = kwargs['data'][usetypes,7]
			else:
				#old output only contains flag, r and type
				self.flag =  kwargs['data'][usetypes,0]
				self.rval = kwargs['data'][usetypes,1:3]
				self.ptype = kwargs['data'][usetypes, 3]
		
				# For defect tracking
				self.vnorm = np.sqrt(self.vval[:,0]**2 + self.vval[:,1]**2+self.vval[:,2]**2)
				self.vhat = self.vval / np.outer(vnorm,np.ones((3,)))
				
				self.N = len(radius)
				self.sigma = np.mean(radius)
				print("New sigma is " + str(self.sigma))

	def gettypes(self, readtypes, frames):
		return np.isin(self.ptype[frames],readtypes)
		
	def truncateto(self,start, endtime):
		self.Nsnap = endtime - start
		self.flag =  self.flag[start:endtime]
		self.rval = self.rval[start:endtime]
		self.vval = self.vval[start:endtime]
		self.theta = self.theta[start:endtime]
		self.nval = self.nval[start:endtime]
		self.radius = self.radius[start:endtime]
		self.ptype = self.ptype[start:endtime]

	def spatialcut(self,minL=-400, maxL=400, dim=0):
		for t in range(self.Nsnap):
			cut_indices = (self.rval[t][:,dim] < minL) | (self.rval[t][:,dim] > maxL)
			self.flag[t][cut_indices] = [0]
			self.vval[t][cut_indices,:] = [0,0]
			self.theta[t][cut_indices] = [0]
			self.nval[t][cut_indices,:] = [0,0]
			self.radius[t][cut_indices] = [0]
			self.ptype[t][cut_indices] = [0]
			self.rval[t][cut_indices,:] = [0,0]