#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Version 3.
Test with known classifications. Ignore the iteration.

======= Nov 13, 2020 =======

另外 为了提高计算速度 我建议你synthetic data就生成一次 估计也就做一次，基于估计的模拟多做几次（十次），
然后把每次的模拟结果和synthetic data比 计算准确度就可以了

其实我们的随机性主要来自于基于估计网络的模拟和最初生成synthetic data的模拟  都是随机的
因此 二者之差也是随机的 如果我们估计准确的话  只对一端做多次模拟求平均 也应该可以剔除一定的随机性的

为了速度 我们先做这个简单的吧

具体到我们的场景就是，

【1 数据阶段】生成一次已知的网络 E，基于这个网络 生成一组diffusion数据 D。

【2 实验阶段】基于数据估计一次网络 E'，然后基于估计网络 E' 反复生成模拟diffusion数据 D'_1, D'_2, D'_3...

【3 评估阶段】把反复生成的多个 diffusion数据 D' 与上面生成的一组真实diffusion数据 D 做对比，对比结果求平均

原则上应该 真实的diffusion数据和模拟的diffusion数据都反复生成 但是只做模拟diffusion反复生成可以提升速度
而且也可以在一定程度上消除随机性

而且如果考虑现实data 而不是这种实验data， 真实的diffusion也只可能有一组  所以我们就先按这个简化版来做

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
from simulation import simulation
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
    today = time.strftime("%Y-%m-%d", time.localtime())
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


    # If you want to do more tests, just add parameters in this list.
    test = [
        [1000, 80],
        [1000, 90],
        [1000, 100],
        [1000, 110],
        [1000, 120],
    ]
    for nodes_num, obs_num in test:
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

        if os.path.exists(save_path + "/result_seed0.log"):
            acc2 = ac.get_accuracy2_cpp(save_path)
            print("Average accuracy2: %.6f" % acc2)
            logging.info("%d x %d Average accuracy2: %.6f" % (nodes_num, obs_num, acc2))
            continue

        is_estimate = True
        files = os.listdir(save_path)
        for f in files:
            if 'to_file_E' in f:
                e_filepath = save_path + "/" + str(f)
                print(e_filepath)
                is_estimate = False
            if 'to_file_true_net_' in f:
                true_net_re_filepath = save_path + "/" + str(f)
                print(true_net_re_filepath)
            if 'obs_' in f and 'original' in f:
                obs_filepath = save_path + "/" + str(f)
                print(obs_filepath)
            if 'true_net_' in f and 'original' in f:
                true_net_filepath = save_path + "/" + str(f)
                print(true_net_filepath)

        if is_estimate:
            # 1 Generate simulation data
            obs_filepath = "/Users/woffee/www/wenbo_at_kong/proposal/data/200x100/obs_200x100_original_11111629.csv"
            true_net_filepath = "/Users/woffee/www/wenbo_at_kong/proposal/data/200x100/true_net_200x100_original_11111629.csv"

            # 2 Estimate the edge matrix E
            e_filepath = "/Users/woffee/www/wenbo_at_kong/proposal/data/200x100/to_file_E_11111628.csv"

            # 3 Process data files
            true_net_re_filepath = "/Users/woffee/www/wenbo_at_kong/proposal/data/200x100/to_file_true_net_11111628_re.csv"


        # 4 Estimate the observation data with E
        acc1 = []
        acc2 = []
        for seed in range(5):
            obs_filepath_2 = save_path + "obs_%dx%d_estimate_seed%d.csv" % (nodes_num, obs_num, seed)
            true_net_filepath_2 = save_path + "true_net_%dx%d_estimate_seed%d.csv" % (nodes_num, obs_num, seed)
            if os.path.exists(obs_filepath_2):
                print("file exists: ", obs_filepath_2)
            else:
                obs_filepath_2, true_net_filepath_2 = sim.do(K, nodes_num, node_dim, ttime, dt, true_net_re_filepath,
                                                             seed)
                logging.info("step 4: " + obs_filepath_2)
                logging.info("step 4: " + true_net_filepath_2)

            # 5 Assess accuracy
            a1 = ac.get_accuracy1(obs_filepath, obs_filepath_2, K, nodes_num)
            acc1.append(a1)
            # a2 = ac.get_accuracy2(obs_filepath, obs_filepath_2, true_net_filepath, true_net_filepath_2, K, nodes_num, save_path)
            filepath_e = save_path + "data_estimate_seed%d.csv" % seed
            filepath_o = save_path + "data_original_seed%d.csv" % seed

            logfile = save_path + "result_seed%d.log" % seed
            ac.build_acc_file(obs_filepath, obs_filepath_2, true_net_filepath, true_net_filepath_2, K, nodes_num, save_path, filepath_o, filepath_e)

            cmd = "/Users/woffee/www/rrpnhat/evaluation/evaluation %s %s >> %s &" % (filepath_e, filepath_o, logfile)
            print(cmd)
            os.system(cmd)

            print("seed:%d, accuracy1:%.6f" % (seed, a1))
            # print("seed:%d, accuracy2:%.6f" % (seed, a2) )
            logging.info("step 5, %d x %d, seed: %d, accuracy1: %.4f" % (nodes_num, obs_num, seed, a1))
            # logging.info("step 5, %d x %d, seed: %d, accuracy2: %.4f" % (nodes_num, obs_num, seed, a2))
        print("Average accuracy1: %.6f" % np.mean(acc1))
        logging.info("%d x %d Average accuracy1: %.6f" % (nodes_num, obs_num, np.mean(acc1)))
        logging.info("---------")
        print()

