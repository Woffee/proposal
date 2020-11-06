"""
Version 3.
Test with known classifications. Ignore the iteration.
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


def ex(x, non_nega_x, threshold):
    x1 = x[non_nega_x]
    x2 = x1.copy()
    x1 = np.exp(x1)
    x1[x2 < np.log(threshold)] = 0.
    y = x.copy()
    y[non_nega_x] = x1
    return y

class MiningHiddenLink:
    def __init__(self, save_path, method_inverse=True, non_nega_cons=[]):
        self.save_path = save_path
        self.method_inverse = method_inverse
        self.non_nega_cons = non_nega_cons

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

    # 有部分x有非负约束的目标函数
    def square_sum_Lagrange_with_ineq(self, x, lbd, D, y):
        # y = np.dot(x,x)
        x = ex(x, self.non_nega_cons, 0.001)
        z = 2 * x - np.dot(D.T, lbd)
        z[self.non_nega_cons] = (2 * x ** 2 - np.dot(D.T, lbd) * x)[self.non_nega_cons]
        z1 = np.dot(D, x) - y

        z = sum(z ** 2) + sum(z1 ** 2)
        return z

    # 有非负约束的梯度函数
    # 其中，部分有非负约束的x的index用self.non_nega_cons这个类属性标注
    def square_sum_Lagrange_with_ineq_grad(self, x, lbd, D, y):
        x = ex(x, self.non_nega_cons, 0.001)
        no_cons = [i for i in range(x.shape[0]) if i not in self.non_nega_cons]
        x_grad = 4 * (2 * x - np.dot(D.T, lbd)) + np.array([sum(2 * (np.dot(D, x) - y) * D[:, i]) for i in range(D.shape[1])]).reshape(D.shape[1], )

        # a1 = 2 * (2 * x ** 2 - np.dot(D.T, lbd) * x)[self.non_nega_cons]
        # a2 = (4 * x ** 2 - np.dot(D.T, lbd) * x)[self.non_nega_cons]
        # a3 = np.array([sum(2 * (np.dot(D, x) - y)[self.non_nega_cons] * D[:, i] * x[i]) for i in self.non_nega_cons]).reshape(len(self.non_nega_cons), )
        # x_grad[self.non_nega_cons] = a1 * a2 + a3

        x_grad[self.non_nega_cons] = 2 * (2 * x ** 2 - np.dot(D.T, lbd) * x)[self.non_nega_cons] * \
                                     (4 * x ** 2 - np.dot(D.T, lbd) * x)[self.non_nega_cons] + np.array(
            [sum(2 * (np.dot(D, x) - y) * D[:, i] * x[i]) for i in self.non_nega_cons]).reshape(
            len(self.non_nega_cons), )

        lbd_grad = -2 * np.dot(D[:, self.non_nega_cons] * x[self.non_nega_cons], (2 * x ** 2 - np.dot(D.T, lbd) * x)[self.non_nega_cons]) - 2 * np.dot(D[:, no_cons], (2 * x - np.dot(D.T, lbd))[no_cons])
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

    def sgd(self, args):
        current_time = datetime.now()
        (theta0, D, y, alpha, iters, delta_min ) = args
        # print("theta0:")
        # print(theta0.shape)
        # print("D:")
        # print(D.shape)
        # print("y:")
        # print(y.shape)

        # (D, y) = args
        m, n = D.shape
        # print(D.shape)
        # exit()

        # Initialize theta and cost history for convergence testing and plot
        theta_hist = np.zeros((iters, theta0.shape[0] + 1))
        theta_hist[0] = np.append(theta0, self.square_sum_Lagrange_with_ineq(theta0[:n], theta0[n:], D, y))

        # Initialize theta generator
        theta_gen = self.grad_adapt(alpha, D, y, self.square_sum_Lagrange_with_ineq_grad)(theta0)

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
        return ex(theta[:n], self.non_nega_cons, 0.001)

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
        r_ma = np.loadtxt(self.save_path + "r_matrix_2.csv", delimiter=',')
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
        #     xit_matrix = []
        #     xit_matrix.append(r_xit)  # y
        #     xit_matrix.append(spreading)  # D
        #     xit_all.append(xit_matrix)
        # edge_list = pool.map(self.minimizer_L1, xit_all)

        m,n = spreading.shape
        x0 =np.array( [1] * n + [0] * m )
        args_all = []
        for y in r_matrix:
            args = (x0, spreading, y, 0.001, 10000, 10**-6)
            args_all.append(args)
        edge_list = pool.map(self.sgd, args_all)
        return np.array(edge_list)

    def do(self, nodes_num, K, obs_num, dt, obs_filepath, true_net_filepath):
        feature_sample, spreading_sample = self.read_data_from_simulation(obs_filepath, true_net_filepath, K,
                                                                        sample_size=nodes_num)
        E = self.get_E(feature_sample, spreading_sample, K, dt)
        E_filepath = self.save_E(E, self.save_path + "to_file_E_" + rundate + ".csv")
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
        [3000, 60],
    ]
    for nodes_num, obs_num in test:
        time = 1.0 * obs_num * dt
        print(nodes_num, obs_num, time)
        logging.info("start: " + str(nodes_num) + "x" + str(obs_num))

        save_path = BASE_DIR + '/data/' + str(nodes_num) + "x" + str(obs_num)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = save_path + "/"

        non_nega_cons = [i * 2 + 1 for i in range(nodes_num)]

        sim = simulation(save_path)
        mhl = MiningHiddenLink(save_path, True, non_nega_cons)
        ac = accuracy(save_path)

        # 1 Generate simulation data
        # obs_filepath, true_net_filepath = sim.do(K, nodes_num, node_dim, time, dt)
        obs_filepath = '/Users/woffee/www/rrpnhat/data/3000x60/obs_3000x60_original_03142022.csv'
        true_net_filepath = '/Users/woffee/www/rrpnhat/data/3000x60/true_net_3000x60_original_03142022.csv'
        logging.info("step 1: " + obs_filepath)
        logging.info("step 1: " + true_net_filepath)


        # 2 Estimate the edge matrix E
        logging.info(str(nodes_num) + "x" + str(obs_num) + "mining hidden link start")
        current_time = datetime.now()
        e_filepath = mhl.do(nodes_num, K, int(time/dt), 0.05, obs_filepath, true_net_filepath)
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
        obs_filepath_2, true_net_filepath_2 = sim.do(K, nodes_num, node_dim, time, dt, true_net_re_filepath)
        logging.info("step 4: " + obs_filepath_2)
        logging.info("step 4: " + true_net_filepath_2)


        # 5 Assess accuracy
        a1 = ac.get_accuracy1(obs_filepath, obs_filepath_2, K, nodes_num)
        print("success rate:", a1)
        logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " accuracy1: " + str(a1))

        # a2 = ac.get_accuracy2(obs_filepath, obs_filepath_2, true_net_filepath, true_net_filepath_2, K, nodes_num)
        # print("accuracy2:", a2)
        # logging.info("step 5 " + str(nodes_num) + "x" + str(obs_num) + " accuracy2: " + str(a2))

