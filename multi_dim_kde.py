from numpy import *
from numba import jit
@jit
def sum_vec(v1,v2):
	return v1+v2
@jit
def gaussiankernel(x,z,args,N):
	if N==1:
		sigma = args
		y = (1./sqrt(2.*pi)/sigma)*exp(-(x-z)**2/(2.*sigma**2))
	else:
		sigma = args
		cov=[]
		for j in range(N):
			cov+=[1./sigma[j,j]**2]
		N=float(N)
		
		y = 1./(2.*pi)**(N/2.)*abs(linalg.det(sigma))**(-1.)*exp((-1./2.)*dot((x-z)**2,array(cov)))
	return y
@jit
def kde2(xdata,xeval,N,ndata,neval,kernel=gaussiankernel,kernelpars=[1.]):
	if N==1:
		if neval>=ndata:
			density = zeros(neval)
			for xj in range(ndata):
				density += kernel(xeval,xdata[xj],kernelpars,N)
				
		else:
			
			density = empty(neval)

			for xi in range(neval):
				density[xi]=kernel(xeval[xi],xdata,kernelpars,N).sum()
				
	else:
		if neval>=ndata:
			density = zeros(neval)
			for xj in range(ndata):
				density += kernel(xeval,xdata[xj,:],kernelpars,N)
		else:
	
			density = empty(neval)
			
			for xi in range(neval):
				
				density[xi]=kernel(xeval[xi,:],xdata,kernelpars,N).sum()
				
	return density/float(ndata)

'''

def density_func(x,x_eval,x_low,x_up,func):
	
	y = array(logical_and(x - x_eval>=zeros_like(x_eval),x - x_eval<=ones_like(x_eval)),dtype = int)
	if (ones(y.shape[0]).dot(y)).dot(ones(y.shape[1]))==0:
		y = array(x - x_eval>=zeros_like(x_eval),dtype = int)
		if (ones(y.shape[0]).dot(y)).dot(ones(y.shape[1]))==0:
			y=x_low
		else: 
	else: pass
	return dot(y[:,0],func)
'''
