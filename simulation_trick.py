# -*- coding: utf-8 -*-
"""
Created on Tue Sep 04 15:16:11 2018
@author: zzxxq
"""
# from threading import Thread
# from queue import Queue
# from numpy import *
# import pandas as pd
# from multi_dim_kde import *
# import datetime
from scipy.stats import norm
from block import *
# from numba import jit
import numpy as np

class network_estimation:
    def __init__(self, time, dt, nodes, val_hidden, trails=500, band_power=0.4, K=2):
        # NOTE: initialize a coup of parameters
        self.time = time  # upper bound of simulation interval
        self.dt = dt  # simulation time interval
        self.nodes = nodes  # nodes of input
        self.val_hidden = val_hidden  # value of hidden network
        self.Vbest = val_hidden  # store the value of estimated hidden network during optimization iteration
        self.val_true = val_hidden  # in simulation study, store the value of treu hidden network
        self.Pbest = None  # likelihood to be updated during iteration in maximization
        self.trails = trails  # max iter num when use stochastic descending algorithm
        self.net2 = None  # hidden network
        self.net1 = None  # composite network
        self.net11 = None  # observable network
        self.band_power = band_power  # kernel bandwidth
        self.K = K

    def simulation(self, val_hidden, nodes, initial, ttime, dt, block_dim, covariates, model_type, net1=None, net2=None,
                   true_net=True, hidden_network_fun=None):
        # NOTE: only applicable to simulation study, simulate the spreading process and generate the time sequence of 0-1 vector of infected status from the given mean-field model
        if model_type == 1:  # normal model with sampled value
            net_fun = lambda a, b, c, d: sample_net(a, b, c, d, evl)
            iter_fun = normal_model
            val_hidden = val_hidden
            coef = array([])
        elif model_type == 2:  # === normal model with values at all nodes ===
            net_fun = norm_net
            iter_fun = normal_model
            val_hidden = val_hidden
            coef = array([])
        elif model_type == 3:  # cluster_model
            net_fun = cluster_model
            iter_fun = normal_model
            val_hidden = val_hidden
            coef = array([])
        elif model_type == 4:  # block_model
            net_fun = block_model
            iter_fun = normal_model
            val_hidden = val_hidden
            coef = array([])
        elif model_type == 5:  # covairate_model
            net_fun = norm_net
            iter_fun = model_with_covariate
            val_hidden = val_hidden[:len(val_hidden) - covariates.shape[1]]
            coef = val_hidden[len(val_hidden) - covariates.shape[1]:]
        else:  # covariate_block_model
            net_fun = block_model
            iter_fun = model_with_covariate
            val_hidden = val_hidden[:len(val_hidden) - covariates.shape[1]]
            coef = val_hidden[len(val_hidden) - covariates.shape[1]:]

        if net1 is None or net2 is None:
            if true_net:
                print('bug here true_net')
                edges = convert_net_to_func(nodes, edges)
                self.edges = edges
                basic_network = network_func(nodes, edges, nodes)
                net1 = basic_network  # generate_network(nodes,basic_network)
            else:
                # here
                net1 = ones((nodes.shape[0], nodes.shape[0]))
            if hidden_network_fun is not None:
                """
                hidden_network_fun == val_hidden
                """
                if callable(hidden_network_fun):
                    print("generate_network")
                    # net2 = generate_network(nodes, hidden_network_fun)
                    net2 = self.normal_distribtution(nodes.shape[0])
                else:
                    net2 = val_hidden.reshape(nodes.shape[0], nodes.shape[0])
                    print(net2)
            else:
                print('bug here hidden_network_fun')
                net2 = net_fun(val_hidden, nodes, block_dim)

            print("net2:", net2.shape)
            self.net2 = net2
            net1 *= net2
            self.net1 = net1
        print("net1:", net1.shape)  # 就是simulation那个nodes
        time_line = append(zeros(1), cumsum(dt * ones(int(ttime / dt))))
        solutions = [initial]
        t = 1
        random.seed(1)
        while t < len(time_line):
            # print 'simulation ',t,'-th run'
            current = solutions[-1]
            # print(current)
            delta = iter_fun(net1, coef, nodes, covariates, current)
            convert_rate = exp(-exp(delta) * dt)
            rand_num = random.rand(len(current))
            solutions.append(current + (rand_num > convert_rate).astype(int))
            t += 1

        # solutions = np.array(solutions).reshape((len(time_line)*self.K, int(net1.shape[0]/self.K) ))
        solutions = np.array(solutions).astype(int)
        print("solutions:", solutions.shape)
        return solutions, time_line

    def normal_distribtution(self, nodes_num):
        print("generate_network normal_distribtution,", nodes_num)
        mu, sigma = 0, 1
        sampleNo = 10000000
        np.random.seed(10)
        rd.seed(10)
        s = np.random.normal(mu, sigma, sampleNo)
        ss = rd.sample(list(s), (nodes_num * nodes_num))
        ss = np.array(ss).reshape(nodes_num, nodes_num)
        return ss

    def minimizer(self, func, v0, iter_num, method1=True):
        ## NOTE: a solve to solve the maximization problem of likelihood function
        ## two solvers are here
        # 1. stochastic descending algorithm, correspond to do_func, which is parallelizable and faster, but less accurate (shown by simulation study)
        # 2. apply scipy built-in solver, 'L-BFGS-B', which is the only built-in solver in scipy that can deal with bounded optimization problem
        self.Vbest = v0
        self.Pbest = None
        jobs = arange(iter_num)

        def do_func1(val):
            P = func(val)
            P, net1, net2 = -P[0], P[1], P[2]
            if self.Pbest is None:
                print('initialization done', P)
                self.Pbest = P
                self.Vbest = val
                self.net1 = net1
                self.net2 = net2
            else:
                print('current', P)

                if P < self.Pbest:
                    print('achieve better', P, self.Pbest, max(absolute(val - self.val_true)))  # ,val,self.val_true,
                    self.Pbest = P
                    self.Vbest = val
                    self.net1 = net1
                    self.net2 = net2
            return P

        def do_func(arguments):
            if arguments == 0:
                val = v0
            else:
                val = norm.cdf(random.randn(len(v0)) + norm.ppf(self.Vbest))
            P = func(val)
            P, net1, net2 = -P[0], P[1], P[2]
            if self.Pbest is None:
                print('initialization done', P)
                self.Pbest = P
                self.Vbest = val
                self.net1 = net1
                self.net2 = net2
            else:
                print('current', arguments, P - self.Pbest, P, val, self.val_true, max(absolute(val - self.val_true)))
                if P < self.Pbest:
                    print('achieve better', arguments, P, self.Pbest, val)
                    self.Pbest = P
                    self.Vbest = val
                    self.net1 = net1
                    self.net2 = net2

        if method1:
            self._start(do_func, jobs, '')
        else:
            from scipy.optimize import minimize
            x0 = v0
            bnds = [(0., 1.) for i in range(v0.shape[0])]
            result = minimize(do_func1, x0, method='L-BFGS-B', bounds=bnds)  # ,options={'maxiter':10000})
        print(result)
