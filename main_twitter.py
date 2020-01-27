"""
Test with twitter data.

Jan 25, 2020
"""

import pandas as pd
import multiprocessing
import numpy as np

from numba import jit
import csv
# import math
# import datetime
import time
# import scipy
import os
import math
# from scipy.optimize import nnls
# from scipy.stats import chi
from scipy.optimize import minimize
from scipy.optimize import Bounds

# import random
# from clean_data import Clean_data
from simulation_twitter import simulation
from accuracy2 import accuracy
import logging
from datetime import datetime


class MiningHiddenLink:
    def __init__(self, save_path):
        self.save_path = save_path

    # calculate error in 1.5
    # def get_min_error(self, E1, E2, n, k, t):
    #     df = (n*k)**2
    #     tmp = (np.std(E1, dtype=np.float64) + np.std(E2,dtype=np.float64)) * 0.5
    #     error = tmp**2 * scipy.stats.chi.ppf(0.9, df) / t
    #     return error


    def is_constrained(self, E1, E2, min_error):
        e = (np.sum( (E1-E2) ** 2 ))
        # print("real_error:",e)
        return e < min_error


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
        print("square_sum: " + str(y))
        logging.info("square_sum: " + str(y))
        return y

    # 1.4
    def minimizer_L1(self, x):
        D=x[1]
        y=x[0].T
        # print('D>>>>>>>>>>>>>>>>>>>>>>')
        # print(D)
        # print('y>>>>>>>>>>>>>>>>>>>>>>')
        # print(y)
        # x0=x[2].reshape(D.shape[1],)-(random.rand(D.shape[1]))/100
        x0=np.ones(D.shape[1],)
        # print('guess x0>>>>>>>>>>')
        # print(x0)
        # print("D:", D.shape)
        # D: (15, 120)
        if(D.shape[0] < D.shape[1]):
            options = {'maxiter': 10, 'ftol': 1e-01, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-02}
            # less observations than nodes
            # see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
            upcons = {'type':'ineq','fun':self.lessObsUpConstrain,'args':(D,y)}
            cur_time = datetime.now()
            result = minimize(self.square_sum, x0, args=(), method='SLSQP', jac=None, bounds=Bounds(0,1),
                              constraints=[upcons], tol=None, callback=None, options=options)
            logging.info("minimizer_L1 time:" + str( datetime.now() - cur_time ) + "," + str(options))
            logging.info("minimizer_L1 result.fun:" + str(result.fun) + ", "
                         + str(result.success) + ", " + str(result.message))
        else:
            logging.info("more observations than nodes")
            result = minimize(self.moreObsfunc, x0, args=(D,y), method='L-BFGS-B', jac=None, bounds=Bounds(0,1), tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
            # print(result)
        return result.x


    # 1.1 & 1.3
    def get_r_xit(self, x, i, t_l, features, spreading, K, bandwidth, dt):
        numerator = 0.0
        denominator = 0.0

        for j in range(features.shape[0]):
            x_j = features.iloc[j]
            g = self.gaussiankernel(x, x_j, bandwidth, features.shape[1])
            tmp = spreading[t_l+1][j*K+i] - spreading[t_l][j*K+i]
            numerator = numerator + (1.0*g*tmp)
            denominator = denominator + (g*dt)
        return numerator/denominator


    def get_r_matrix(self, features, spreading, K=2, dt=0.01):
        bandwidth = np.diag(np.ones(features.shape[1]) * float(features.shape[0]) ** (-1. / float(features.shape[1] + 1)))

        r_matrix = []
        for x in range(features.shape[0]):
            print("get_r_matrix now x:",x)
            for i in range(K):
                row = []
                for t in range(spreading.shape[0] - 1):
                    r_xit = self.get_r_xit(features.iloc[x], i, t, features, spreading, K, bandwidth, dt)
                    row.append(r_xit)
                r_matrix.append(row)
                # print(row)

        return np.array(r_matrix)


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
        # print("clear 0:")
        # print(res)

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

        # print("r_matrix: ", r_matrix.shape)
        logging.info("r_matrix.shape: " + str( r_matrix.shape))

        # spreading = spreading[subT, :]
        spreading = np.delete(spreading, deleted, axis=0)
        spreading = np.delete(spreading, -1, axis=0)

        logging.info("features.shape:" + str(features.shape))
        logging.info("spreading.shape:" + str(spreading.shape))
        # logging.info("subT:" + str(subT))

        # np.savetxt(save_path + 'to_file_spreading_' + rundate + ".txt", spreading)
        # np.savetxt(save_path + 'to_file_r_matrix_' + rundate + ".txt", r_matrix)

        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)


        # spreading = np.delete(spreading, -1, axis=0)
        # spreading = clear_zeros(spreading)

        xit_all = []
        for r_xit in r_matrix:
            xit_matrix = []
            xit_matrix.append(r_xit)  # y
            xit_matrix.append(spreading)  # D
            xit_all.append(xit_matrix)
        edge_list = pool.map(self.minimizer_L1, xit_all)

        return np.array(edge_list)

    def do(self, nodes_num, K, obs_num, dt, obs_filepath, feature_filepath):
        feature_sample = pd.read_csv(feature_filepath, header=None)
        spreading_sample = pd.read_csv(obs_filepath, header=None)

        # feature_sample = np.array(feature_sample)
        spreading_sample = np.array(spreading_sample).T

        E = self.get_E(feature_sample, spreading_sample, K, dt)
        E_filepath = self.save_E(E, self.save_path + "to_file_E_.csv")
        logging.info(E_filepath)
        return E_filepath

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    rundate = time.strftime("%m%d%H%M", time.localtime())
    # to_file = save_path + "to_file_" + rundate + ".csv"
    today = time.strftime("%Y-%m-%d", time.localtime())
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=BASE_DIR + '/log/' + today + '.log')

    K = 2
    nodes_num = 50
    obs_num = 30

    node_dim = 2
    dt = 0.05


    ttime = 1.0 * obs_num * dt
    print(nodes_num, obs_num, ttime)
    logging.info("start: " + str(nodes_num) + "x" + str(obs_num))

    save_path = BASE_DIR + '/data_twitter/'
    mhl = MiningHiddenLink(save_path)
    ac = accuracy(save_path)
    sim = simulation(save_path)


    obs_filepath = save_path + 'obs.csv'
    feature_filepath = save_path + 'features.csv'

    # 1 Estimate the edge matrix E
    logging.info("twitter mining hidden link start")
    current_time = datetime.now()
    e_filepath = mhl.do(nodes_num, K, int(ttime/dt), 0.05, obs_filepath, feature_filepath)
    logging.info("twitter mining hidden link done")
    logging.info("twitter mining hidden link time: " + str( datetime.now() - current_time ))
    print(e_filepath)

    # 2 Process data files
    true_net_filepath = save_path + "true_net.csv"
    true_net = pd.read_csv(true_net_filepath, sep=',')

    true_net_re_filepath = save_path + "true_net_re.csv"
    hidden_link = pd.read_csv(e_filepath, sep=',', header=None)
    true_net['e'] = hidden_link.values.flatten()
    true_net.to_csv(true_net_re_filepath, header=True, index=None)
    logging.info("step 3: " + true_net_re_filepath)

    # 3 Estimate the observation data with E
    obs_filepath_2, true_net_filepath_2 = sim.do(K, nodes_num, node_dim, ttime, dt, true_net_re_filepath, obs_filepath)
    logging.info("step 4: " + obs_filepath_2)
    logging.info("step 4: " + true_net_filepath_2)

    # 4 Assess accuracy
    a1 = ac.get_accuracy1(obs_filepath, obs_filepath_2, K, nodes_num)
    a2 = ac.get_accuracy2(obs_filepath, obs_filepath_2, true_net_filepath, true_net_filepath_2, K, nodes_num)
    print("accuracy1:", a1)
    print("accuracy2:", a2)
    logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " accuracy1: " + str(a1))
    logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " accuracy2: " + str(a2))




