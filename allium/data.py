import numpy as np
class Parameters(object):
	def __init__(self, p):
		for key, values in p.items():
			setattr(self, key, values)

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
		class param:
			def __init__(self):
				self.Lx = 800
				self.Ly = 800
				self.R = 8
				self.dt = 0.01
				self.output_time = 83

		self.param = param()
		self.sigma = 8
		self.umpp = 0.8
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


		self.vval = np.zeros((properties['t'][-1]-2,maxNcells,2))
		for n in range(0, properties['t'][-1]-3):
			self.vval[n+1,self.ptype[n+1] == 1,:]  = (self.rval[n+1][self.ptype[n+1] == 1] - self.rval[n][self.ptype[n] == 1])*(self.param.output_time*self.param.dt )


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
