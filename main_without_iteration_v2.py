#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Version 3.
Test with known classifications. Ignore the iteration.

适合节点数大的情况。
"""

import pandas as pd
import multiprocessing
import numpy as np

# from numba import jit
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
from simulation2.simulation import simulation
from accuracy2 import accuracy
import logging
from datetime import datetime
import cvxpy as cp
import argparse

r_matrix_result_list = []

# 1.4
# In particular, functions are only picklable if they are defined at the top-level of a module.
def minimizer_L1(x):
    D=x[1]
    y=x[0].T
    # x0=np.ones(D.shape[1],)
    # if(D.shape[0] < D.shape[1]):
    #     options = {'maxiter': 10, 'ftol': 1e-01, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-02}
    #     # less observations than nodes
    #     # see https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html#optimize-minimize-slsqp
    #     upcons = {'type':'ineq','fun':self.lessObsUpConstrain,'args':(D,y)}
    #     cur_time = datetime.now()
    #     result = minimize(self.square_sum, x0, args=(), method='SLSQP', jac=None, bounds=Bounds(0,1),
    #                       constraints=[upcons], tol=None, callback=None, options=options)
    #     logging.info("minimizer_L1 time:" + str( datetime.now() - cur_time ) + "," + str(options)
    #                  + " result.fun:" + str(result.fun) + ", " + str(result.success) + ", " + str(result.message))
    # else:
    #     logging.info("more observations than nodes")
    #     result = minimize(self.moreObsfunc, x0, args=(D,y), method='L-BFGS-B', jac=None, bounds=Bounds(0,1), tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
    #     # print(result)


    x_val = None
    scale = 1
    while x_val is None and scale < 16:
        x = cp.Variable(D.shape[1])
        objective = cp.Minimize(cp.sum(cp.abs(x)) / (float(D.shape[1]) ** scale))

        constraints = [D * x - y.flatten() == 0]

        prob = cp.Problem(objective, constraints)
        # ECOS, OSQP, SCS
        result = prob.solve(verbose=True, solver='SCS', feastol=1e-9)
        x_val = x.value

        scale += 0.1

    if x_val is None:
        print("x.value is None. scale:" + str(scale))
    # print 'optimal val :', result
    # return np.append(np.zeros(pshape), x.value)  # .reshape(self.pshape+D.shape[1],1)
    return x.value


def gaussiankernel(x, z, args, N):
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

# 1.1 & 1.3
def get_r_xit(x, i, t_l, features, spreading, K, T, bandwidth, dt):
    numerator = 0.0
    denominator = 0.0
    # print(features.shape)
    for j in range(features.shape[0]):
        x_j = features.iloc[j]
        g = gaussiankernel(x, x_j, bandwidth, features.shape[1])
        tmp = spreading[t_l+1][j*K+i] - spreading[t_l][j*K+i]
        numerator = numerator + (1.0*g*tmp)
        denominator = denominator + (g*(T[t_l+1]-T[t_l])*dt)
    return numerator/denominator

def log_result(result):
    # This is called whenever foo_pool(i) returns a result.
    # result_list is modified only by the main process, not the pool workers.
    r_matrix_result_list.append(result)

def get_r_row(args):
    # features.iloc[x], i, features, spreading, K, T, bandwidth, dt
    index, fea, i, features, spreading, K, T, bandwidth, dt = args
    row = []
    for t in range(len(T) - 1):
        r_xit = get_r_xit(fea, i, t, features, spreading, K, T, bandwidth, dt)
        row.append(r_xit)
    return [index, row]



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



    def read_data_from_simulation(self, obs_filepath, true_net_filepath, K, sample_size = 100):
        data = pd.read_csv(true_net_filepath, encoding='utf-8')

        ## get the features of nodes ##
        feature_sample = data[['node1_x', 'node1_y']]
        features = feature_sample.drop_duplicates()
        features.index = np.arange(sample_size)

        spreading_sample = pd.read_csv(obs_filepath, encoding='utf-8')
        # spreading_sample.drop('user_id', axis=1, inplace=True)
        # spreading_sample.drop([spreading_sample.columns[0]], axis=1, inplace=True)
        # spreading = spreading_sample.values
        spreading_sample = np.array(spreading_sample)

        # obs = spreading[::10] # [开始：结束：步长]
        # print(obs.shape) # (101, 600)
        # features_matirx = features.values

        index = range(len(spreading_sample))
        deleted = []
        # for i in range(len(spreading_sample)):
        #     if i > 0:
        #         last = spreading_sample[i - 1]
        #         now = spreading_sample[i]
        #         if (last == now).all():
        #             deleted.append(i)
        # print("deleted:", deleted)
        T = list(set(index).difference(set(deleted)))
        # print(T)
        # print("features_matirx:",features.shape)
        # print("spreading_sample:",spreading_sample.shape)
        return features, spreading_sample, T




    def get_r_matrix(self, features, spreading, T, K=2, dt=0.01):
        bandwidth = np.diag(np.ones(features.shape[1]) * float(features.shape[0]) ** (-1. / float(features.shape[1] + 1)))
        start = time.time()

        # 单线程 版本
        # r_matrix = []
        # for x in range(features.shape[0]):
        #     print("get_r_matrix now x:",x)
        #     for i in range(K):
        #         row = []
        #         for t in range(len(T) - 1):
        #             r_xit = get_r_xit(features.iloc[x], i, t, features, spreading, K, T, bandwidth, dt)
        #             row.append(r_xit)
        #         r_matrix.append(row)

        cores = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=cores)
        print("get_r_matrix, cores:%d" % cores)
        logging.info("get_r_matrix, cores:%d" % cores)

        for x in range(features.shape[0]):
            # print("get_r_matrix now x:",x)
            for i in range(K):
                pool.apply_async( get_r_row, ([ x*2 + i, features.iloc[x], i, features, spreading, K, T, bandwidth, dt ],) , callback = log_result)

        pool.close()
        pool.join()

        end = time.time()
        logging.info("time for r_matrix: {} seconds".format((end - start)))
        print("get r_matrix done")


        global r_matrix_result_list
        r_matrix_result_list = sorted(r_matrix_result_list, key=lambda k: k[0] )
        r_matrix = [ row[1] for row in r_matrix_result_list ]

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
    def get_E(self, features, spreading, subT, K, dt=0.01):
        logging.info("dt:" + str(dt))
        r_matrix_path = self.save_path + "r_matrix.csv"
        # print("dt:",dt)
        if os.path.exists(self.save_path + "r_matrix.csv"):
            print("loading r_matrix_path:", r_matrix_path)
            logging.info("loading r_matrix_path: " + r_matrix_path)
            r_matrix = np.loadtxt(r_matrix_path, delimiter=',')
        else:
            r_matrix = self.get_r_matrix(features, spreading, subT, K, dt)
            np.savetxt(self.save_path + "r_matrix.csv", r_matrix, delimiter=',')
            print("saved to r_matrix_path:", r_matrix_path)
            logging.info("saved to r_matrix_path: " + r_matrix_path)


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
        edge_list = pool.map(minimizer_L1, xit_all)

        return np.array(edge_list)

    def do(self, nodes_num, K, obs_num, dt, obs_filepath, true_net_filepath):
        feature_sample, spreading_sample, T = self.read_data_from_simulation(obs_filepath, true_net_filepath, K,
                                                                        sample_size=nodes_num)
        E = self.get_E(feature_sample, spreading_sample, T, K, dt)
        E_filepath = self.save_E(E, self.save_path + "to_file_E_" + rundate + ".csv")
        logging.info(E_filepath)
        return E_filepath

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    rundate = time.strftime("%m%d%H%M", time.localtime())
    today = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    default_log_file = BASE_DIR + '/' + today + '.log'


    parser = argparse.ArgumentParser(description='Test for argparse')
    parser.add_argument('--log', '-l', help='log', default= default_log_file )
    parser.add_argument('--nodes_num', '-n', help='nodes_num', type=int, default=100)
    parser.add_argument('--obs_num', '-o', help='obs_num', type=int, default=80)
    args = parser.parse_args()



    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=args.log)

    print(args)

    K = 2
    nodes_num = args.nodes_num
    obs_num = args.obs_num

    node_dim = 2
    dt = 0.05

    ttime = 1.0 * obs_num * dt
    print(nodes_num, obs_num, ttime)
    logging.info("start: " + str(nodes_num) + "x" + str(obs_num))

    save_path = BASE_DIR + '/data/' + str(nodes_num) + "x" + str(obs_num)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    save_path = save_path + "/"

    sim = simulation(save_path)
    mhl = MiningHiddenLink(save_path)
    ac = accuracy(save_path)

    # 1 Generate simulation data
    obs_filepath, true_net_filepath = sim.do(K, nodes_num, node_dim, ttime, dt)
    logging.info("step 1: " + obs_filepath)
    logging.info("step 1: " + true_net_filepath)

    # 2 Estimate the edge matrix E
    logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link start")
    current_time = datetime.now()
    e_filepath = mhl.do(nodes_num, K, int(ttime/dt), 0.05, obs_filepath, true_net_filepath)
    logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link done")
    logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link time: " + str( datetime.now() - current_time ))
    logging.info("step 2: " + e_filepath)

    # 3 Process data files
    true_net_re_filepath = save_path + "to_file_true_net_" + rundate + "_re.csv"
    true_net = pd.read_csv(true_net_filepath, sep=',')
    hidden_link = pd.read_csv(e_filepath, sep=',', header=None)
    true_net['e'] = hidden_link.values.flatten()
    true_net.to_csv(true_net_re_filepath, header=True, index=None)
    logging.info("step 3: " + true_net_re_filepath)

    # 4 Estimate the observation data with E
    obs_filepath_2, true_net_filepath_2 = sim.do(K, nodes_num, node_dim, ttime, dt, true_net_re_filepath)
    logging.info("step 4: " + obs_filepath_2)
    logging.info("step 4: " + true_net_filepath_2)

    # 5 Assess accuracy
    a1 = ac.get_accuracy1(obs_filepath, obs_filepath_2, K, nodes_num)
    a2 = ac.get_accuracy2(obs_filepath, obs_filepath_2, true_net_filepath, true_net_filepath_2, K, nodes_num)
    print("accuracy1:", a1)
    print("accuracy2:", a2)
    logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " accuracy1: " + str(a1))
    logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " accuracy2: " + str(a2))

