"""
Version 3.
Test with known classifications. Ignore the iteration.

对算法和simulation的代码 做了两处修改 修改后的框架下 可以彻底抛掉部分x有非负约束的设定
所以可以直接使用显示解方法进行计算

果结果ok 我们直接进入后续的工作 可以暂时不考虑sgd方法

请用后一份main1.py和上面的simulation_trick.py在跑一边 我已经把一切设定都调好了（可能有语法错误，你需要看一下），
另外，记得一定要重新生成simulation data，因为simulation的关键一步做了相应的调整，如果不重新生成，可能存在比较大的估计偏误
"""

import pandas as pd
import multiprocessing
import numpy as np

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
	def __init__(self, save_path, method_inverse=True, non_nega_cons=[], all_non_nega_cons=False):
		self.save_path = save_path
		self.method_inverse = method_inverse
		self.non_nega_cons = non_nega_cons
		self.all_non_nega_cons = all_non_nega_cons

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
				print('{} minimum change in theta not achieved in {} iterations.'
					  .format(delta_min, theta_hist.shape[0]))
				break
			delta = np.max(np.square(theta - theta_hist[i - 1, :-1])) ** 0.5

			i += 1
		# Trim zeros and return
		theta_hist = theta_hist[:i]
		print("finished: %d, time: %s" % (i, str(datetime.now() - current_time)) )
		return theta[:n]

	def explicit_minimizer(self, args):
		(D,y)=args
		# print("D:", D.shape)
		# print("y:", y.shape)
		# y = np.dot(x,x)
		DD = np.append(np.diag(np.ones(D.shape[1]) * 2), D.T, axis=1)
		DD = np.append(DD, np.append(D, np.zeros((D.shape[0], D.shape[0])), axis=1), axis=0)
		# print("DD:", DD.shape)
		y = np.append(np.zeros(D.shape[1]), y, axis=0)
		x = np.dot(np.linalg.inv(DD), y)
		# print("x:", x.shape)
		return x[:D.shape[1]]

	


	def read_data_from_simulation(self, obs_filepath, true_net_filepath, K, sample_size = 100):
		data = pd.read_csv(true_net_filepath, encoding='utf-8')

		## get the features of nodes ##
		feature_sample = data[['node1_x', 'node1_y']]
		features = feature_sample.drop_duplicates()
		features.index = np.arange(sample_size)

		spreading_sample = pd.read_csv(obs_filepath, encoding='utf-8')
		spreading_sample = np.array(spreading_sample)

		index = range(len(spreading_sample))
		deleted = []

		# T = list(set(index).difference(set(deleted)))
		return features, spreading_sample


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


	def get_r_matrix(self, features, spreading, K=2, dt=0.01):
		if os.path.exists(self.save_path + "r_matrix.csv"):
			r_ma = np.loadtxt(self.save_path + "r_matrix.csv", delimiter=',')
			print(r_ma.shape)
			return r_ma

		bandwidth = np.diag(np.ones(features.shape[1]) * float(features.shape[0]) ** (-1. / float(features.shape[1] + 1)))

		current_time = datetime.now()
		r_matrix = []
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
		res = np.array(r_matrix)
		logging.info("r_matrix time: " + str(datetime.now() - current_time))
		np.savetxt(self.save_path + "r_matrix.csv", res, delimiter=',')
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


	# 1.1 & 1.4
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
		x0 =np.array( [1] * n + [0] * m )
		args_all = []
		for y in r_matrix:
			dele=np.where(y==0)[0]
			z=np.log(np.delete(y, dele, axis=0))
			spd=np.delete(spreading, dele, axis=0)
			spd=spd/float(spd.shape[1])
			#args = (x0, spd, z, 0.001, 10000, 10**-6)
			args = (spd, z)
			args_all.append(args)
			
		edge_list = pool.map(self.explicit_minimizer, args_all)
		#edge_list = pool.map(self.sgd, args_all)
		return np.array(edge_list)

	def do(self, nodes_num, K, obs_num, dt, obs_filepath, true_net_filepath):
		feature_sample, spreading_sample = self.read_data_from_simulation(obs_filepath, true_net_filepath, K,
																		sample_size=nodes_num)
		E = self.get_E(feature_sample, spreading_sample, K, dt)
		E_filepath = self.save_E(E, self.save_path + "to_file_E.csv")
		logging.info(E_filepath)
		return E_filepath

if __name__ == '__main__':
	BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	save_path = BASE_DIR + '/data/'
	rundate = time.strftime("%m%d%H%M", time.localtime())
	# to_file = save_path + "to_file_" + rundate + ".csv"
	today = time.strftime("%Y-%m-%d", time.localtime())
	logging.basicConfig(level=logging.INFO,
						format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
						datefmt='%Y-%m-%d %H:%M:%S',
						filename=BASE_DIR + '/log/' + today + '.log')

	K = 2
	nodes_num = 100
	node_dim = 2
	time = 7.5
	dt = 0.05

	# If you want to do more tests, just add parameters in this list.
	test = [
		[500, 100],
	]
	for nodes_num, obs_num in test:
		time = 1.0 * obs_num * dt
		print(nodes_num, obs_num, time)
		logging.info("start: " + str(nodes_num) + "x" + str(obs_num))

		save_path = BASE_DIR + '/data/' + str(nodes_num) + "x" + str(obs_num)
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		save_path = save_path + "/"

		non_nega_cons = [i for i in range(nodes_num * K)]

		sim = simulation(save_path)
		mhl = MiningHiddenLink(save_path, True, non_nega_cons, True)
		ac = accuracy(save_path)

		# 1 Generate simulation data
		obs_filepath = save_path + 'obs_' + str(nodes_num) + "x" + str(obs_num) + '_original.csv'
		true_net_filepath = save_path + 'true_net_' + str(nodes_num) + "x" + str(obs_num) + '_original.csv'
		if not os.path.exists(obs_filepath) or not os.path.exists(true_net_filepath):
			obs_filepath, true_net_filepath = sim.do(K, nodes_num, node_dim, time, dt)
		else:
			print("simulation data existed")
		logging.info("step 1: Generate simulation data done")

		# 2 Estimate the edge matrix E
		e_filepath = save_path + "to_file_E.csv"
		if not os.path.exists(e_filepath):
			logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link start")
			current_time = datetime.now()
			e_filepath = mhl.do(nodes_num, K, int(time/dt), 0.05, obs_filepath, true_net_filepath)
			logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link done")
			logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link time: " + str( datetime.now() - current_time ))
		else:
			print("E existed")
		logging.info("step 2: " + e_filepath)

		# 3 Process data files
		true_net_re_filepath = save_path + "to_file_true_net_re.csv"
		if not os.path.exists(true_net_re_filepath):
			true_net = pd.read_csv(true_net_filepath, sep=',')
			hidden_link = pd.read_csv(e_filepath, sep=',', header=None)
			true_net['e'] = hidden_link.values.flatten()
			true_net.to_csv(true_net_re_filepath, header=True, index=None)
		logging.info("step 3: " + true_net_re_filepath)

		# 4 Estimate the observation data with E
		obs_filepath_2 = save_path + 'obs_' + str(nodes_num) + "x" + str(obs_num) + '_estimate.csv'
		true_net_filepath_2 = save_path + 'true_net_' + str(nodes_num) + "x" + str(obs_num) + '_estimate.csv'
		if not os.path.exists(obs_filepath_2) or not os.path.exists(true_net_filepath_2):
			sim.do(K, nodes_num, node_dim, time, dt, true_net_re_filepath)
		else:
			print("estimated data existed")
		logging.info("step 4: estimate data done")


		# 5 Assess accuracy
		a1 = ac.get_accuracy1(obs_filepath, obs_filepath_2, K, nodes_num)
		print("success rate:", a1)
		logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " success rate: " + str(a1))

		a3 = ac.get_accuracy3(obs_filepath, obs_filepath_2)
		print("deviation:", a3)
		logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " deviation: " + str(a3))

		a2 = ac.get_accuracy2(obs_filepath, obs_filepath_2, true_net_filepath, true_net_filepath_2, K, nodes_num)
		print("accuracy:", a2)
		logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " accuracy: " + str(a2))



