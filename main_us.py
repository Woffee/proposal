"""
Version 3.
Test with known classifications. Ignore the iteration.
"""

import pandas as pd
import multiprocessing
import numpy as np
from numpy import *
from numba import jit
import csv
import time
import os
import math
from scipy.optimize import minimize
from scipy.optimize import Bounds
from simulation import simulation
from accuracy2 import accuracy
import logging
from datetime import datetime


class MiningHiddenLink:
	def __init__(self, save_path, method_inverse=True, fixed_input=None,non_nega_cons=[], all_non_nega_cons=False):
		self.save_path = save_path
		self.method_inverse = method_inverse
		self.non_nega_cons = non_nega_cons
		self.all_non_nega_cons = all_non_nega_cons
		self.fixed_input=fixed_input
		if fixed_input is not None:
			self.pshape=fixed_input.shape[0]
	@jit
	def gaussiankernel(self, x, z, args, N):
		if N == 1:
			sigma = args
			y = (1. / math.sqrt(2. * math.pi) / sigma) * math.exp(-(x - z) ** 2 / (2. * sigma ** 2))
		else:
			sigma = args
			cov = []
			for j in range(N):
				cov += [1. / sigma[j, j] ** 2]
			N = float(N)

			y = 1. / (2. * math.pi) ** (N / 2.) * abs(np.linalg.det(sigma)) ** (-1.) * math.exp(
				(-1. / 2.) * np.dot((x - z) ** 2, np.array(cov)))
		return y


	def lasso(self, x, D, y):
		temp = (np.dot(D,x))/D.shape[1] - y
		eq = np.dot(temp,temp)
		return eq+np.dot(x,x)


	def lessObsUpConstrain(self, x, D, y):
		temp = (np.dot(D,x))/D.shape[1] - y
		eq = np.dot(temp,temp)
		return -eq+0.1


	def moreObsfunc(self, x, D, y):
		temp = y.reshape(len(y),1)-np.dot(D,x.reshape(len(x),1))
		temp = temp.reshape(1,len(temp))
		return np.asscalar(np.dot(temp,temp.T))


	def square_sum(self, x):
		# y = np.dot(x,x)
		y = np.sum( x**2 )
		return y

	# 1.4
	def minimizer_L1(self, x):
		# D: (M, N)
		D=x[1]
		y=x[0].T
		x0=np.ones(D.shape[1],)
		if(D.shape[0] < D.shape[1]):
			# less observations than nodes
			# Adjust the options' parameters to speed up when N >= 300
			# see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
			options = {'maxiter': 10, 'ftol': 1e-01, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-02}
			upcons = {'type':'ineq','fun':self.lessObsUpConstrain,'args':(D,y)}
			cur_time = datetime.now()
			result = minimize(self.square_sum, x0, args=(), method='SLSQP', jac=None, bounds=Bounds(0,1),
							  constraints=[upcons], tol=None, callback=None, options=options)
			# logging.info("minimizer_L1 time:" + str( datetime.now() - cur_time ) + "," + str(options) + " result.fun:" + str(result.fun) + ", " + str(result.success) + ", " + str(result.message))
		else:
			logging.info("more observations than nodes")
			result = minimize(self.moreObsfunc, x0, args=(D,y), method='L-BFGS-B', jac=None, bounds=Bounds(0,1), tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
		return result.x




	# 没有非负约束的目标函数
	def square_sum_Lagrange(self, x, lbd, D, y):
		# y = np.dot(x,x)
		z = 2 * x - np.dot(D.T, lbd)
		z1 = np.dot(D, x) - y
		z = sum(z ** 2) + sum(z1 ** 2)
		return z

	# 没有非负约束的梯度函数
	def square_sum_Lagrange_grad(self, x, lbd, D, y):
		# y = np.dot(x,x)
		# tt = np.array( [sum(2 * (np.dot(D, x) - y) * D[:, i]) for i in range(D.shape[1])]).reshape(D.shape[1],)
		# print(tt.shape)
		x_grad = 4 * (2 * x - np.dot(D.T, lbd)) + np.array(
			[sum(2 * (np.dot(D, x) - y) * D[:, i]) for i in range(D.shape[1])]).reshape(D.shape[1], )
		lbd_grad = -2 * np.dot(D, (2 * x - np.dot(D.T, lbd)))
		# print("111")
		# print(x_grad.shape)
		# print(lbd_grad.shape)
		# print(x_grad)
		return np.append(x_grad, lbd_grad, axis=0)



	# Initialize gradient adaptation.
	def grad_adapt(self, alpha, D, y, grad_fun):
		(m, n) = D.shape

		def theta_gen_const(theta):
			while True:
				theta = theta - alpha * grad_fun(theta[:n], theta[n:], D, y)
				# print(theta)
				yield theta

		return theta_gen_const
	
	def grad_adapt_ineq(self, alpha, D, y, grad_fun):
		(m, n) = D.shape

		def theta_gen_const(theta):
			while True:
				grad=grad_fun(theta[:n], theta[n:], D, y)
				non_nega_grad=grad[self.non_nega_cons]
				theta_non_nega=theta[self.non_nega_cons]
				indicator=np.where((non_nega_grad>=0)*(theta_non_nega<=0))[0]
				grad[np.array(self.non_nega_cons,dtype=int)[indicator]]=0
				theta = theta - alpha * grad
				nt=theta[self.non_nega_cons]
				theta[np.where(nt<0)[0]]=0
				# print(theta)
				yield theta

		return theta_gen_const

	def sgd(self, args,ineq=False):
		current_time = datetime.now()
		(theta0, D, y, alpha, iters, delta_min ) = args
		m, n = D.shape
		
		# Initialize theta and cost history for convergence testing and plot
		theta_hist = np.zeros((iters, theta0.shape[0] + 1))
		theta_hist[0] = np.append(theta0, self.square_sum_Lagrange(theta0[:n], theta0[n:], D, y))

		# Initialize theta generator
		if ineq:
			self.grad_adapt_ineq(alpha, D, y, self.square_sum_Lagrange_grad)(theta0)
		else:
			theta_gen = self.grad_adapt(alpha, D, y, self.square_sum_Lagrange_grad)(theta0)

		# Initialize iteration variables
		delta = float("inf")
		i = 1

		theta = theta0
		# Run algorithm
		while delta > delta_min:
			# Get next theta
			theta = next(theta_gen)
			# print(theta)
			# Store cost for plotting, test for convergence
			try:
				cost = self.square_sum_Lagrange_with_ineq(theta[:n], theta[n:], D, y)
				if cost > theta_hist[i - 1][-1]:
					break
				theta_hist[i] = np.append(theta, cost)
			except:
				print '{} minimum change in theta not achieved in {} iterations.'
					  .format(delta_min, theta_hist.shape[0])
				break
			delta = np.max(np.square(theta - theta_hist[i - 1, :-1])) ** 0.5

			i += 1
		# Trim zeros and return
		theta_hist = theta_hist[:i]
		print "finished: %d, time: %s" % (i, str(datetime.now() - current_time)) 
		return theta[:n]
	def explicit_minimizer(self, args):
		if len(args)==2:
			(D,y)=args
			# y = np.dot(x,x)
			DD = np.append(np.diag(np.ones(D.shape[1]) * 2), D.T, axis=1)
			DD = np.append(DD, np.append(D, np.zeros((D.shape[0], D.shape[0])), axis=1), axis=0)
			y = np.append(np.zeros(D.shape[1]), y, axis=0)
		else:
			(D,y,params,K,cshape)=args
			DD = np.append(np.diag(np.ones(D.shape[1]) * 2), D.T, axis=1)
			y = np.append(np.zeros(D.shape[1]), y, axis=0)
			for i in range(K):
				if i==0:
					DD[arange(self.pshape),arange(self.pshape)]=DD[arange(self.pshape),arange(self.pshape)]+2
					y[:self.pshape]=2*params[:self.pshape]
				else:
					DD[self.pshape+(i-1)*cshape:self.pshape+(i)*cshape][:,self.pshape+(i-1)*cshape:self.pshape+(i)*cshape]=DD[self.pshape+(i-1)*cshape:self.pshape+(i)*cshape][:,self.pshape+(i-1)*cshape:self.pshape+(i)*cshape]+2
					y[self.pshape+(i-1)*cshape:self.pshape+(i)*cshape]=y[self.pshape+(i-1)*cshape:self.pshape+(i)*cshape]+2*params[self.pshape+i-1]
			DD = np.append(DD, np.append(D, np.zeros((D.shape[0], D.shape[0])), axis=1), axis=0)
			
		return np.dot(np.linalg.inv(DD), y)

	


	def read_data_from_simulation(self, obs_filepath):
		

		spreading_sample = pd.read_csv(obs_filepath, encoding='utf-8')
		spreading_sample = np.array(spreading_sample)

		

		# T = list(set(index).difference(set(deleted)))
		return spreading_sample


	# 1.1 & 1.3
	def get_r_xit(self, x, i, t_l, features, spreading, K, bandwidth, dt, G):
		numerator = 0.0
		denominator = 0.0
		# print(features.shape)
		for j in range(features.shape[0]):
			# x_j = features.iloc[j]
			# g = self.gaussiankernel(x, x_j, bandwidth, features.shape[1])
			tmp = spreading[t_l+1][j*K+i] - spreading[t_l][j*K+i]
			numerator = numerator + (1.0 * G[j] * tmp)
			denominator = denominator + (G[j] * dt)
		return numerator/denominator


	def get_r_matrix(self, features, spreading):
		# r_ma = np.loadtxt(self.save_path + "r_matrix.csv", delimiter=',')
		# print(r_ma.shape)
		# return r_ma

		#bandwidth = np.diag(np.ones(features.shape[1]) * float(features.shape[0]) ** (-1. / float(features.shape[1] + 1)))

		current_time = datetime.now()
		r_matrix = []
		r_matrix=spreading[1:]-spreading[:-1]
		'''
		for x in range(features.shape[0]):
			print("get_r_matrix now x:",x)
			G = [] # 这里存一下每个节点x与其他节点的K_h值
			for j in range(features.shape[0]):
				G.append(self.gaussiankernel(features.iloc[x], features.iloc[j], bandwidth, features.shape[1]))

			for i in range(K):
				row = []
				for t in range(spreading.shape[0]-1):
					r_xit = self.get_r_xit(features.iloc[x], i, t, features, spreading, K, bandwidth, dt, G)
					row.append(r_xit)
				r_matrix.append(row)
				# print(row)
		'''
		res = np.array(r_matrix)
		
		logging.info("r_matrix time: " + str(datetime.now() - current_time))
		np.savetxt(self.save_path + "r_matrix_2.csv", res, delimiter=',')
		return res


	def save_E(self, E, filepath):
		# print(E1)
		# print("E:", len(E), len(E[0]))
		with open(filepath, "w") as f:
			writer = csv.writer(f)
			writer.writerows(E)
		# print(filepath)
		return filepath

	def clear_zeros(self, mitrix):
		# delete t with all zeros
		all_zero_columns = np.where(~mitrix.any(axis=0))[0]
		res = np.delete(mitrix, all_zero_columns, axis=1)
		return res


	
	'''
	def get_E(self, features, spreading, K, dt=0.01):
		logging.info("dt:" + str(dt))
		# print("dt:",dt)
		r_matrix = self.get_r_matrix(features, spreading, K, dt)
		
		sum_col = np.sum(r_matrix, axis=0)
		deleted = []
		for i in range(len(sum_col)):
			if sum_col[i] == 0:
				deleted.append([i])
		r_matrix = np.delete(r_matrix, deleted, axis=1) # delete columns where all 0
		logging.info("r_matrix_deleted:" + str(deleted))
		logging.info("r_matrix.shape: " + str( r_matrix.shape))

		spreading = np.delete(spreading, deleted, axis=0)
		spreading = np.delete(spreading, -1, axis=0)
		cshape=int(spreading.shape[1]/K)
		spd=[]
		rmax=[]
		for i in range(K):
			if i==0:
				spd=spreading[:,0::K]
				rmax=rmax[i::K,:]
			else:
				spd=np.append(spd,spreading[:,i::K],axis=1)
				rmax=np.append(rmax,spreading[i::K,:],axis=0)
		spreading=spd.copy()
		spd=np.append(np.zeros((7,spd.shape[1])),spd,axis=0)
		spd=spd[7:]-spd[:-7]
		spreading[:,chspae:]=spd[:,cshape:]
		r_matrix=rmax.copy()
		logging.info("features.shape:" + str(features.shape))
		logging.info("spreading.shape:" + str(spreading.shape))

		cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=cores)

		# xit_all = []
		# for r_xit in r_matrix:
		#	 xit_matrix = []
		#	 xit_matrix.append(r_xit)  # y
		#	 xit_matrix.append(spreading)  # D
		#	 xit_all.append(xit_matrix)
		# edge_list = pool.map(self.minimizer_L1, xit_all)

		m,n = spreading.shape
		x0 =np.diag(np.ones( cshape ))
		it=0
		while it<=itmax:
			args_all = []
			ny=0
			for y in r_matrix:
				#dele=np.where(y<=0)[0]
				if ny<cshape:
					z=y-np.dot(x0[ny],spreading[:,:cshape].T)
					spd=spreading[:,cshape:]
					spd=spd/float(spd.shape[1])
					args = (spd, z)
					args_all.append(args)
				else:
					z=np.log(1+y)[1:]-np.log(1+y)[:-1]
					#spd=np.delete(spreading, dele, axis=0)
					spd=spreading[1:]
					spd=spd/float(spd.shape[1])
					#args = (x0, spd, z, 0.001, 10000, 10**-6)
					args = (spd, z)
					args_all.append(args)
				ny+=1
			edge_list = pool.map(self.explicit_minimizer, args_all)
			ny=0
			for y in r_matrix:
				if ny<cshape:
					z=y[:,:cshape]-np.dot(edge_list[ny],spreading[:,cshape:].T)
					spd=spreading[:,:cshape]
					args = (spd, z,x0)
					args_all.append(args)
				ny+=1
			infect_list = pool.map(self.explicit_minimizer, args_all)
			x0=np.array(infect_list)
			for i in range(len(edge_list)):
				if i <cshape:
					edge_list[i]=np.append(x0[i],edge_list[i],axis=0)
					
			edge_list=np.array(edge_list)
			if it==0:
				out=edge_list.copy()
			else:
				diffe=sum(edge_list-out)**2/(sum(edge_list**2+out**2)/2.)
				out=edge_list.copy()
				if diffe<0.05:
					break
			it+=1
		#edge_list = pool.map(self.sgd, args_all)
		return np.array(edge_list)
	'''
	# 1.1 & 1.4
	def get_E(self, spreading,K):
		
		r_matrix = self.get_r_matrix(None, spreading, K)
		
		sum_col = np.sum(r_matrix, axis=0)
		deleted = []
		for i in range(len(sum_col)):
			if sum_col[i] == 0:
				deleted.append([i])
		r_matrix = np.delete(r_matrix, deleted, axis=1) # delete columns where all 0
		logging.info("r_matrix_deleted:" + str(deleted))
		logging.info("r_matrix.shape: " + str( r_matrix.shape))

		spreading = np.delete(spreading, deleted, axis=0)
		spreading = np.delete(spreading, -1, axis=0)
		cshape=int((spreading.shape[1]-self.pshape)/(K-1))
		'''
		spd=[]
		rmax=[]
		
		for i in range(K):
			if i==0:
				spd=spreading[:,0::K]
				rmax=rmax[i::K,:]
			else:
				spd=np.append(spd,spreading[:,i::K],axis=1)
				rmax=np.append(rmax,spreading[i::K,:],axis=0)
		spreading=spd.copy()
		
		#spd=np.append(np.zeros((7,spd.shape[1])),spd,axis=0)
		#spd=spd[7:]-spd[:-7]
		#spreading[:,chspae:]=spd[:,cshape:]
		r_matrix=rmax.copy()
		'''
		#logging.info("features.shape:" + str(features.shape))
		logging.info("spreading.shape:" + str(spreading.shape))

		cores = multiprocessing.cpu_count()
		pool = multiprocessing.Pool(processes=cores)

		# xit_all = []
		# for r_xit in r_matrix:
		#	 xit_matrix = []
		#	 xit_matrix.append(r_xit)  # y
		#	 xit_matrix.append(spreading)  # D
		#	 xit_all.append(xit_matrix)
		# edge_list = pool.map(self.minimizer_L1, xit_all)

		m,n = spreading.shape
		
		iter=0
		
		while iter<self.iter_max:
			args_all = []
			ny=0
			
			for y in r_matrix:
				#dele=np.where(y<=0)[0]
				if ny<self.pshape:
					z=self.fixed_input[ny]
					spd=spreading[:,self.pshape:]
					spd=spd/float(spd.shape[1])
					args = (spd, z)
					args_all.append(args)
				else:
					z=np.log(1+y)[1:]-np.log(1+y)[:-1]
					#spd=np.delete(spreading, dele, axis=0)
					spd=spreading[1:]
					spd=spd/float(spd.shape[1])
					if iter==0:
						
						#args = (x0, spd, z, 0.001, 10000, 10**-6)
						args = (spd, z)
						
					else:
						ind=(ny-self.pshape)/cshape
						args=(spd,z,uni_params[ind],K,cshape)
					args_all.append(args)	
				ny+=1
			edge_list = pool.map(self.explicit_minimizer, args_all)
			univ=[]
			for i in range(1,K):
				if i>=1:#self.pshape:
					av=mean(array(edge_list[self.pshape+(i-1)*cshape:self.pshape+(i)*cshape]),axis=0)
					avv=av[:self.pshape]
					for j in range(K-1)
						avv=append(avv,ones(1)*sum(av[self.pshape+(j-1)*cshape:self.pshape+(j)*cshape]))
					univ.append(avv)
			if iter==0:
				
				uni_params=array(univ)
			else:
				univ=array(univ)
				if sum((univ-uni_params)**2)/min([sum(univ**2),sum(uni_params**2)])<0.05:
					uni_params=univ
					break
				else:
					uni_params=univ
			iter+=1
		for i in range(self.pshape):
			edge_list[i]=np.append(zeros(cshape),edge_list[i],axis=0)
		for i in range(self.pshape,len(edge_list)):
			edge_list[i]=np.append(uni_params[(i-self.pshape)/cshape,:self.pshape],edge_list[i,self.pshape:],axis=0)
			for j in range(K-1):
				edge_list[i,self.pshape+(j-1)*cshape:self.pshape+(j)*cshape]=edge_list[i,self.pshape+(j-1)*cshape:self.pshape+(j)*cshape]*uni_params[(i-self.pshape)/cshape,self.pshape+j]/sum(edge_list[i,self.pshape+(j-1)*cshape:self.pshape+(j)*cshape])
		#edge_list = pool.map(self.sgd, args_all)
		return np.array(edge_list)
	def do(self, nodes_num, K, infect_data,spreading_sample, media_data):
		spreading_sample=append(infect_data,media_data,axis=1)
		E = self.get_E( spreading_sample, K)
		E_filepath = self.save_E(E, self.save_path + "to_file_E_" + rundate + ".csv")
		logging.info(E_filepath)
		return E_filepath
def reg(infect_matrix,popu_data,incub1,incub,os_T,date_index,infect_prob):
	x,y=[],[]
	for i in range(len(infect_matrix)):
		if os_T*(i)+incub1+incub<=len(date_index):
			d_index=array(date_index)[len(date_index)-incub1-incub-os_T*i:len(date_index)-os_T*i]
		
		else:
			d_index=array(date_index)[:len(date_index)-os_T*i]
		x.append(moving_average(popu_data,infect_prob[-i-1],incub,d_index))
	x=x[::-1]
	
	im=[infect_matrix[i] for i in range(len(infect_matrix))]
	inm=array(im).flatten()
	xx=array(x,dtype=float).flatten()
	
	xx=array([ones_like(xx),xx],dtype=float).T
	#print dot(xx.T,xx)
	coef=linalg.inv(dot(xx.T,xx)).dot(dot(xx.T,inm))
	#sumsquare=lambda coef:reduce(ssm,[sum((coef[0]+coef[1]*x[i]-infect_matrix[i][1:,1:])**2) for i in range(len(infect_matrix))[start:end]])
	#res=minimize(sumsquare,x0=zeros(2))
	#print res
	#xx=array(x[:7]).flatten()
	
	#print xx.shape,inm.shape
	#coef=sum((xx-mean(xx))*(inm-mean(inm)))/float(xx.shape[0])/std(xx)/std(inm)
	#coef1=res.x
	print 'R^2', coef#,sumsquare(coef)/float(len(inm))/std(inm)**2#coef1,res.fun/sum(inm**2),
	

	
	return coef
def moving_average(data,prob,incub,d_index,starting=None,theta=None,):
	matrix=zeros_like(data[d_index[0]])
	a=zeros(data[d_index[0]].shape[0])
	b=zeros(data[d_index[0]].shape[0])
	
	for s in range(len(d_index))[incub:]:
		
		dd=zeros_like(matrix)
		
		for i,k in enumerate(d_index[s-incub:s]):
			
						
			dd=dd+data[k]*prob[i]
			
		
		matrix=matrix+dd
	matrix=matrix/float(len(data)-incub)
	
	return matrix#dot(a,matrix)#matrix/float(len(data)-incub)
def counterfact_sim(coef,popu_data,infect_prob,incub1,incub,os_T,date_index):
	
	popu=popu_data.copy()
	
	
	for i in range(len(infect_matrix)):
		if os_T*(i+1)+incub<=len(date_index):
			d_index=array(date_index)[len(date_index)-incub-os_T*i:len(date_index)-os_T*i]
			d1_index=array(date_index)[len(date_index)-incub-os_T*i-1:len(date_index)-os_T*i-1]
		
		else:
			d_index=array(date_index)[:len(date_index)-os_T*i]
			d1_index=array(date_index)[:len(date_index)-os_T*i-1]
		a=moving_average(popu,infect_prob[-i-1],incub,d_index)
		
		y=a*coef[1]+coef[0]
		x.append(y)
		b=moving_average(popu,infect_prob[-i-1],incub,d1_index)
		
		y=b*coef[1]+coef[0]

		
		x.append(y)
		
	x=x[::-1]	
	return x	
	
if __name__ == '__main__':
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	save_path = BASE_DIR + '/data/'
	rundate = time.strftime("%m%d%H%M", time.localtime())
	
	
	incub=14
	incub1=7
	os_T=2
	##read and process infect/hiddent infect data
	infect_prob=[]
	recovery=[]
	infect_matrix=[]
	boom_prob=[]
	hidden_infect=[]
	raw_data=[]
	for i in range(8):
		data=pd.ExcelFile('result_'+str(i)+'.xlsx')
		item ='ad_matrix_0'
		infect_matrix.append(array(data.parse(item))[:,1:])
		item='prob_0'
		infect_prob.append(array(data.parse(item))[:,2])
		boom_prob.append(array(data.parse(item))[:,1])
		item='recovery_0'
		recovery.append(array(data.parse(item))[0,1])
		item='hidden_data_0'
		hd=data.parse(item)
		hidden_infect.append(array(hd[[it for it in hd.columns if 'pre' in str(it)]]))
		item='raw_data_0'
		raw_data.append(array(data.parse(item))[:,1:])
	raw_data=raw_data[::-1]
	cities=array(data.parse('recovery_0'))[:,0:1]
	
	infect_matrix=infect_matrix[::-1]
	infect_prob=infect_prob[::-1]
	boom_prob=boom_prob[::-1]
	recovery=recovery[::-1]
	hidden_infect=hidden_infect[::-1]
	
	popu=pd.read_csv('movement matrix_county_03-08_to04-09_new.csv')
	popu.index=popu['origin_fips/destination_fips']
	popu.fillna(0,inplace=True)
	popu=array(popu.ix[cities[:,0]][cities[:,0]])[:,1:].T
	date_index=['2020-03-0'+str(i) for i in range(8,10)]+['2020-03-'+str(i) for i in range(10,32)]
	date_index=date_index+['2020-04-0'+str(i) for i in range(1,10)]
	popu_data={it:popu for it in date_index}
	
	coef=reg(infect_matrix,popu_data,incub1,incub,os_T,date_index,infect_prob)
	fitted_matrix=counterfact_sim(coef,popu_data,infect_prob,incub1,incub,os_T,date_index)
	for i in range(8):
		
		
		if i==0:
			hidden_data=hidden_infect[i]
			report_data=raw_data[i]
			
		else:
			
			
			report_data=append(report_data,raw_data[i][:,-os_T:],axis=1)
			hidden_data=append(hiddent_data,hidden_infect[i][:,-os_T:],axis=1)
	hidden=[]
	hidden_data=hidden_data[1:]
	hidden=hidden_data[:,incub+1:]-hidden_data[:,incub:-1]
	forecast_hidden=[]
	h=hidden.shape[1]
	f=h#len(fitted_matrix)
	for j in range(f):
		o1=append(hidden[:,h-f+j]-dot(dot(fitted_matrix[j],hidden_data[:,h-f+j:h-f+j+incub]),infect_prob[int(j/2)])+hidden_data[:,h-f+j+incub]*recovery[int(j/2)][0])
		forecast_hidden.append(o1)
	fixed_input=array(forecast_hidden)
	infect_data=report_data[:,:f].T
	##calculate the increment of hidden infect
	
	# to_file = save_path + "to_file_" + rundate + ".csv"
	today = time.strftime("%Y-%m-%d", time.localtime())
	logging.basicConfig(level=logging.INFO,
						format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
						datefmt='%Y-%m-%d %H:%M:%S',
						filename=BASE_DIR + '/log/' + today + '.log')

	K = 2
	nodes_num = 100
	node_dim = 2
	#time = 7.5
	#dt = 0.05

	# If you want to do more tests, just add parameters in this list.
	#test = [
	#	[500, 100],
	#]
	for nodes_num, obs_num in test:
		#time = 1.0 * obs_num * dt
		print nodes_num, obs_num
		logging.info("start: " + str(nodes_num) + "x" + str(obs_num))

		save_path = BASE_DIR + '/data/' + str(nodes_num) + "x" + str(obs_num)
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		save_path = save_path + "/"

		non_nega_cons = [i for i in range(nodes_num * K)]

		#sim = simulation(save_path)
		mhl = MiningHiddenLink(save_path, True,fixed_input, non_nega_cons, True)
		ac = accuracy(save_path)

		# 1 Generate simulation data
		#obs_filepath, true_net_filepath = sim.do(K, nodes_num, node_dim, time, dt)
		# obs_filepath = '/Users/woffee/www/rrpnhat/data/500x100/obs_500x100_original_03172311.csv'
		# true_net_filepath = '/Users/woffee/www/rrpnhat/data/500x100/true_net_500x100_original_03172311.csv'
		#logging.info("step 1: " + obs_filepath)
		#logging.info("step 1: " + true_net_filepath)
		obs_filepath ='obs_k1.csv'
		media_data=pd.read_csv(obs_filepath)
		media_data=array(media_data[list(cities[:,0])])[:f]
		# 2 Estimate the edge matrix E
		logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link start")
		current_time = datetime.now()
		e_filepath = mhl.do(nodes_num, K, infect_data, media_data)
		logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link done")
		logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link time: " + str( datetime.now() - current_time ))
		logging.info("step 2: " + e_filepath)
'''
		# 3 Process data files
		true_net_re_filepath = save_path + "to_file_true_net_" + rundate + "_re.csv"
		true_net = pd.read_csv(true_net_filepath, sep=',')
		hidden_link = pd.read_csv(e_filepath, sep=',', header=None)
		true_net['e'] = hidden_link.values.flatten()
		true_net.to_csv(true_net_re_filepath, header=True, index=None)
		logging.info("step 3: " + true_net_re_filepath)

		# 4 Estimate the observation data with E
		obs_filepath_2, true_net_filepath_2 = sim.do(K, nodes_num, node_dim, time, dt, true_net_re_filepath)
		logging.info("step 4: " + obs_filepath_2)
		logging.info("step 4: " + true_net_filepath_2)


		# 5 Assess accuracy
		a1 = ac.get_accuracy1(obs_filepath, obs_filepath_2, K, nodes_num)
		print "success rate:", a1
		logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " accuracy1: " + str(a1))

		# a2 = ac.get_accuracy2(obs_filepath, obs_filepath_2, true_net_filepath, true_net_filepath_2, K, nodes_num)
		# print("accuracy2:", a2)
		# logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " accuracy2: " + str(a2))
'''
