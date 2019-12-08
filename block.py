# -*- coding: utf-8 -*-

# from threading import Thread
# from queue import Queue
# from numpy import *
# import pandas as pd
from multi_dim_kde import *
# import datetime
# import time
# from scipy.stats import norm
from numba import jit,generated_jit
rd=True
@jit
def block_model(val_hidden,nodes,block_dim):
	prob=val_hidden[:nodes.shape[0]*block_dim].reshape(block_dim,nodes.shape[0])
	block_matrix=val_hidden[nodes.shape[0]*block_dim:].reshape(block_dim,block_dim)
	dens=kde2(nodes,nodes,nodes.shape[1],nodes.shape[0],nodes.shape[0],kernelpars=diag(ones(nodes.shape[1])*float(nodes.shape[0])**(-1./float(nodes.shape[1]+1))))
	prob=[]
	for j in range(block_dim):
		prob1=[]
		for i in range(nodes.shape[0]):
			prob1+=[mean(prob[j]*gaussiankernel(nodes[i],nodes,args=diag(ones(nodes.shape[1])*float(nodes.shape[0])**(-1./float(nodes.shape[1]+1))),N=nodes.shape[1])/dens)]
		prob+=[prob1]
	prob=array(prob)
	prob=dot(prob.T,block_matrix).dot(prob)
	return prob

@jit
def cluster_model(val_hidden,nodes,block_dim):
	prob=val_hidden[:nodes.shape[0]*block_dim].reshape(block_dim,nodes.shape[0])
	print(len(val_hidden),nodes.shape[0]*block_dim)
	prob[prob-amax(prob,axis=0)<0]=0
	block_matrix=val_hidden[nodes.shape[0]*block_dim:].reshape(block_dim,block_dim)
	#dens=kde2(nodes,nodes,nodes.shape[1],nodes.shape[0],nodes.shape[0],kernelpars=diag(ones(nodes.shape[1])*float(nodes.shape[0])**(-1./float(nodes.shape[1]+1))))
	#Prob=[]
	#for j in range(block_dim):
	#	prob1=[]
	#	for i in range(nodes.shape[0]):
	#		prob1+=[mean(prob[j]*gaussiankernel(nodes[i],nodes,args=diag(ones(nodes.shape[1])*float(nodes.shape[0])**(-1./float(nodes.shape[1]+1))),N=nodes.shape[1])/dens)]
	#	Prob+=[prob1]
	#prob=array(Prob)	
	prob=dot(prob.T,block_matrix).dot(prob)
	return prob


@jit
def ss(a,b):
	return a+b
@jit
def model_with_covariate(net,coef,nodes,covariates,iter_var):
	prob=net
	coef=coef
	coef1=exp(dot(covariates,coef))
	new=array([])
	for i in range(covariates.shape[0]):
		a=coef1[i]*prob
		b=iter_var[i*nodes,shape[0]:(i+1)*nodes,shape[0]]
		c=dot(a,b)
		for j in range(covariates.shape[0]):
			if j != i:
				c+=dot(prob,iter_var[j*nodes.shape[0]:(j+1)*nodes.shape[0]])
		 
		new=append(new,c)
	return new/float(len(iter_var))	
@jit
def normal_model(net,coef,nodes,covariates,iter_var):
	#print net.shape,iter_var.shape
	new=dot(net,iter_var)
	return new/float(len(iter_var))
@jit
def normal_model_randv(net,coef,nodes,covariates,iter_var):
	new=[]
	for i in range(iter_var.shape[0]):
		k=random.choice(arange(nodes.shape[0]))
		b=iter_var[k]
		new+=[net[i,k]*b]
	return array(new)
@jit
def model_with_covariate_randv(net,coef,nodes,covariates,iter_var):
	prob=net
	coef=coef
	coef1=exp(dot(covariates,coef))
	coef1=exp()
	new=array([])
	for i in range(iter_var.shape[0]):
		k,l=random.choice(arange(nodes.shape[0])),random.choice(arange(covariates.shape[0]))
		
		b=iter_var[i]
		if int(i/(k+1))==l:
			a=coef1[l]*prob[i,k]
		else:
			a=prob[i,k]
		c=a*b
		
		new=append(new,ones(1)*c)
	return new	


@jit
def network_func(data_point,binary_val,inp):
	if len(binary_val.shape)==1:
		print("=============== 08-3-1")
		data=data_point
		val=binary_val
		inp=inp
	else:
		print("=============== 08-3-2")
		data=[]
		val=[]
		for i in range(data_point.shape[0]):
			a=data_point[i]
			a1=binary_val[i]
			for j in range(data_point.shape[0]):
				b=data_point[j]
				b1=a1[j]
				data.append(append(a,b))
				val.append(b1)
		data=array(data)
		val=array(val)
		inp1=[]
		for i in range(inp.shape[0]):
			a=inp[i]
			
			for j in range(inp.shape[0]):
				b=inp[j]
				
				inp1+=[append(a,b)]
				
		inp=array(inp1)
		
	# NOTE: apply lo	cal constant estimator to generate empirical edge function 
	dens=kde2(data,inp,data.shape[1],data.shape[0],inp.shape[0],kernelpars=diag(ones(data.shape[1])*float(data.shape[0])**(-1./float(data.shape[1]+1))))
	func1=[]
	#print dens
	for i in range(inp.shape[0]):
	
		func1+=[mean(val*gaussiankernel(inp[i],data,args=diag(ones(data.shape[1])*float(data.shape[0])**(-1./float(data.shape[1]+1))),N=data.shape[1])/dens[i])]
	return array(func1)    
@jit	
def convert_net_to_func(nodes,edges):
	## NOTE: convert the nodes-edges specification of a network to its nodes-adjacancy matrix specification ##
	func=[]
	for i in range(nodes.shape[0]):
		func1=[]
		for j in range(nodes.shape[0]):
			val=0
			ind=[]
			for item in edges:
				ind+=[i==item[0]]
			ind=where(array(ind))[0]
			ind1=[]
			for item in array(edges)[ind]:
				ind1+=[j==item[1]]
			ind=where(array(ind1))[0]		
			val+=len(ind)
			ind=[]
			for item in edges:
				ind+=[i==item[1]]
			ind=where(array(ind))[0]
			ind1=[]
			for item in array(edges)[ind]:
				ind1+=[j==item[0]]
			ind=where(array(ind1))[0]
			
			val+=len(ind)
			print(val)
			func1.append(val)
		func.append(func1)
	return array(func)
@jit
def generate_network(data_point,network):
	net1=[]      
	for i in range(data_point.shape[0]):
		net11=[] 
		for j in range(data_point.shape[0]):
			net11.append(network(append(data_point[i],data_point[j])))
		net1.append(net11)
	net1=array(net1)
	return net1
@jit
def sample_net(val_hidden,nodes,block_dim,evl):
	hidden_network=network_func(evl,val_hidden,nodes)
	#net2=generate_network(nodes,hidden_network)
	return net2


# here is net_fun
@jit
def norm_net(val_hidden,nodes,block_dim):
	net2=val_hidden.reshape(nodes.shape[0],nodes.shape[0])
	return net2
@jit
def solve_ode(initial,net1,covar,covariates,data_point,time,dt,iter_fun,rd):
	## NOTE: numerically solve the mean-field equation and generate pdf of observations ##
	time_line=append(zeros(1),cumsum(dt*ones(int(time/dt))))
	solutions=[initial]
	t=1
	while t<len(time_line):
		current=solutions[-1]
		if len(current[current>1.])>0:
			print('something wrong1',where(current>1.),current[current>1.])
			if len(current[current>1.0000000000001])>0:
				quit()
		delta=(1-current)*iter_fun(net1,covar,data_point,covariates,current)
		current=current+delta*dt
		if len(current[current<0.])>0.:
			print('something wrong2',where(current<0.))
			quit()
		current[current>=1.]=1.
		#if rd:
		#	rand=random.rand(len(current))
		#	current=(rand<=current).astype(float)
		solutions+=[current]
		t+=1
	return solutions,time_line
    

@jit    	
def likelihood(obs,obs_t,val_hidden,net1,evl,nodes,nodes1,initial,time,dt,covariates,net_fun,cov_dim,iter_fun,block_dim,rd=True):
	## NOTE: construct likelihood function from simulation and observation ##
	# nodes are a set of nodes whose infection stauts are observable at all obs_t, nodes are represented as an n x d dimensional matrix, n is the number of nodes, d is the dimension of features of node
	# nodes1 are a set of nodes to evaluate the edge weight, nodes1 are represented as an m x d dimensional matrix, m is the number of nodes, m<n,    
	# because the continuous asssumption of edge functions, we can use a smaller set of node pairs (nodes1) to evaluate the edge function. and then apply kernel density method to generate an appraoximate edge function, and evaluate it on a larger set of node pairs (nodes), in this way, THE COMPUTATION SPEED CAN BE LIFTED, AS A MUCH LOWER DIMENSIONLA PARAMETER VECTOR ARE NEEDED IN OPTIMIZATION!
	############################################################
	#two types of observation input#
	#1. follow-up observation of one spreading processes:
	# obs is a list of 0-1 valued vector encoding with the info whether a node is infected, for each vector, its dimension is equal to nodes.shape[0]
	# obs_t is a set of observation time point, its length should be equal to the length of obs
	# initial is a 0-1 valued vector represeted as the infection situation of nodes, its length should be equal to nodes1.shape[0]
	#2. a sequence of follow-up observation of multiple spreading processes:
	# obs is a list of obs of type 1
	# obs_t is a list of obs_t of type 1
	# initial is a list of initial of type 1
	#######################
	#two types of edge input#
	#1. evl,val_hidden pair#
	# evl is a set of node pairs, n x 2d dimensional matrix, n is the number of pairs, d is the dimension of node features
	# val_hidden is a 1d array of weights that are associated with nodes pairs evl
	#2. square matrix val_hidden#
	# for this type of input, val_hidden is supposed to be a square matrix, its dimension is equal to nodes1.shape[0]
	###################################################################
	# block_model
	net2=net_fun(val_hidden[:val_hidden.shape[0]-cov_dim],nodes1,block_dim)
	net11=net1
	net12=net1*net2
	covar=val_hidden[val_hidden.shape[0]-cov_dim:]
	print("=============== 07")
	print("=============== 07 len of obs_t:",len(obs_t))
	if type(initial)!=list:
		#NOTE: apply to observation input of type 1
		#generate pdf of infected status of every node in nodes1 by simulate the mean-field equation following the method in the paper sec 3.2, 3.3
		val,time_line=solve_ode(initial,net12,covar,covariates,nodes1,time,dt,iter_fun,rd)
		ind=index(time_line,obs_t)
		print("=============== 08 len of ind:", len(ind))
		Prob=[]
        
		for k,i in enumerate(ind):
			# NOTE: the observed infected status is with respect to nodes, so convert the pdf of being infected at every node in nodes1 to the pdf at every node in nodes by applying the kernel method 
			print("=============== 08")
			print("=============== 08 nodes1:", nodes1.shape)
			prob=network_func(nodes1,val[i],nodes)#val[i]
			realization=obs[k]
			prob[realization==0]=1-prob[realization==0]
			if any(isnan(log(prob))):
				print('why nan1:',val[i][val[i]<0])
				print('why nan2:',prob[isnan(log(prob))])
			print('infected:',len(prob[prob==0]),len(prob[(prob==0)*(realization==0)]),len(prob))
			prob[prob==0]=1.e-30
			prob=mean(log(prob))
			Prob.append(prob)
		Prob=sum(Prob)
	else:
		# NOTE: apply to observation input of type 2
		PP=[]
		for i in range(len(initial)):
			print("=============== 08-2 len of initial:", len(initial))
			val,time_line=solve_ode(initial[i],net12,covar,covariates,nodes1,time,dt,iter_fun,rd)
			ind=index(time_line,obs_t[i])
			print("=============== 08-2 len of ind:", len(ind))
			Prob=[]
			for k,j in enumerate(ind):
				print('zeros_prob1:',len(val[j][val[j]==0]))
				# print("=============== 08-2")
				print("=============== 08-3 nodes1:", nodes1.shape)
				print("=============== 08-3 nodes:", nodes.shape)
				prob=network_func(nodes1,val[j],nodes)
				print('zeros_prob2:',len(prob[prob==0]))
				realization=obs[i][k]
				prob[realization==0]=1-prob[realization==0]
				print('infected:',len(prob[prob==0]),len(prob[(prob==0)*(realization==0)]),len(realization[realization==0]),len(prob))
				prob[prob==0]=1.e-30
				if any(isnan(log(prob))):
					print('why nan1:',val[j][val[j]<0])
					print('why nan2:',prob[isnan(log(prob))])
				prob=mean(log(prob))
				Prob.append(prob)
				print("=============== 08-3 end")
			Prob=sum(Prob)
			PP.append(Prob)
		Prob=sum(PP)
	return [Prob,net12,net2]
@jit
def index(time_line,to_be_convert):
	# NOTE: when a time point in obs_t is not contained in the time_line generated from method 'solve_ode', select the time point in time_line that is closest to the given observation time
	time_line=outer(time_line,ones_like(to_be_convert))
	time_line-=to_be_convert
	return argmin(absolute(time_line),axis=0)
@jit
def simulation(self,val_hidden,edges,nodes,evl,initial,time,dt,block_dim,covariates,model_type,net1=None,net2=None,true_net=True,hidden_network_fun=None):
	# NOTE: only applicable to simulation study, simulate the spreading process and generate the time sequence of 0-1 vector of infected status from the given mean-field model
	if model_type==1:#normal model with sampled value
		net_fun=lambda a,b,c,d:sample_net(a,b,c,d,evl)
		iter_fun=normal_model
		val_hidden=val_hidden
		coef=array([])
	elif model_type==2:#normal model with values at all nodes
		net_fun=norm_net
		iter_fun=normal_model
		val_hidden=val_hidden
		coef=array([])
	elif model_type==3:#cluster_model
		net_fun=cluster_model
		iter_fun=normal_model
		val_hidden=val_hidden
		coef=array([])
	elif model_type==4:#block_model
		net_fun=block_model
		iter_fun=normal_model
		val_hidden=val_hidden
		coef=array([])
	elif model_type==5:#covairate_model
		net_fun=norm_net
		iter_fun=model_with_covariate
		val_hidden=val_hidden[:len(val_hidden)-covariates.shape[1]]
		coef=val_hidden[len(val_hidden)-covariates.shape[1]:]
	else:#covariate_block_model
		net_fun=block_model
		iter_fun=model_with_covariate	
		val_hidden=val_hidden[:len(val_hidden)-covariates.shape[1]]
		coef=val_hidden[len(val_hidden)-covariates.shape[1]:]
		
	if net1 is None or net2 is None:
		if true_net:        
			edges=convert_net_to_func(nodes,edges)
			self.edges=edges
			basic_network=network_func(nodes,edges,nodes)
			net1=basic_network#generate_network(nodes,basic_network)
		else:
			net1=ones((nodes.shape[0],nodes.shape[0]))
		if hidden_network_fun is not None:
			net2=generate_network(nodes,hidden_network_fun)
		else:
			net2=net_fun(val_hidden,nodes,block_dim)
		
	time_line=append(zeros(1),cumsum(dt*ones(int(time/dt))))
	solutions=[initial]
	t=1
	while t<len(time_line):
		print('simulation ',t,'-th run')
		current=solutions[-1]
		delta=(1-current)*iter_fun(net1,coef,nodes,covariates,current)
		convert_rate=exp(-delta*dt)
		rand_num=random.rand(len(current))
		solutions.append(current+(rand_num>convert_rate).astype(int)*(current==0).astype(int))
		t+=1
	return solutions,time_line
 
