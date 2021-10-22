#use all <- doesn't look like it works... 
def getVelcorrSingle(data,dx,xmax,whichframe=1,usetype='all',verbose=True):
	# start with the isotropic one - since there is no obvious polar region
	# and n is not the relevant variable, and v varies too much
	print("Velocity correlation function for frame " + str(whichframe))
	npts=int(round(xmax/dx))
	bins=np.linspace(0,xmax,npts)
	velcorr=np.zeros((npts,))
	velcount=np.zeros((npts,))
	#index relevant particles (by default we use all of them)
	useparts = data.gettypes(usetype,whichframe)
	N = sum(useparts)
	velav=np.sum(data.vval[whichframe,useparts,:],axis=0)/N
	for k in range(N):
		vdot=np.sum(data.vval[whichframe,useparts[k],:]*data.vval[whichframe,useparts,:],axis=1)

		# problem probably here		
		#ApplyPeriodicBC and take norm
		dr=np.linalg.norm(ApplyPeriodic2d(data, data.vval[whichframe,useparts,:] - data.vval[whichframe,useparts[k],:]))


		drbin=(np.round(dr/dx)).astype(int)
		for l in range(npts):
			pts=np.nonzero(drbin==l)[0]
			velcorr[l]+=vdot[pts].sum()
			velcount[l]+=len(pts)
			
	isdata=[index for index, value in enumerate(velcount) if value>0]
	velcorr[isdata]=velcorr[isdata]/velcount[isdata] - np.sum(velav*velav)
	if verbose:
		fig=plt.figure()
		isdata=[index for index, value in enumerate(velcount) if value>0]
		plt.plot(bins[isdata],velcorr[isdata],'.-r',lw=2)
		#plt.show()
		plt.xlabel("r-r'")
		plt.ylabel('Correlation')
		plt.title('Spatial velocity correlation')
	return bins,velcorr


