import numpy as np
class Parameters(object):
	def __init__(self, p):
		for key, values in p.items():
			setattr(self, key, values)

class Data:
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
			self.N = 0
			Nvals = []
			for t in range(self.start,self.end):
				Nvals.append(len(Data.checkTypes(self.readtypes, kwargs['data'][t])))
				if Nvals[t] > self.N:
					self.N = Nvals[t] 

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
				usetypes = Data.checkTypes(self.readtypes, kwargs['data'][t])
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
					self.sigma = np.mean(kwargs['data'][t][usetypes,6])
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
			usetypes = Data.checkTypes(self.readtypes, kwargs['data'][t])
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
		
