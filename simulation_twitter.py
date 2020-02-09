# -*- coding: utf-8 -*-
"""
Created on Aug Sep 04 17:22:43 2019

@author: zxq
"""

# from numpy import *
import time as t_module
import pandas as pd
from simulation_trick import network_estimation
# from scipy.stats import chi2
from block import *
# import sys
import os

class simulation:
    def __init__(self, save_path):
        self.save_path = save_path

    def do(self, K, nodes_num, node_dim, ttime, dt, filepath = None, obs_filepath=None, test_num=1):
        if filepath and os.path.exists(filepath):
            file_net = pd.read_csv(filepath)
            nodes = file_net[['node2_1','node2_2','node2_3','node2_4','node2_5']].iloc[range(nodes_num * K)].values
            val_hidden = file_net['e'].values
        else:
            exit()

        # 2. max possible edges counts, gererate those index
        evl = []
        for k in range(nodes.shape[0]):
            for j in range(nodes.shape[0]):
                evl.append(append(nodes[k], nodes[j]))
        evl = array(evl)

        hidden_net = []
        obs = pd.read_csv(obs_filepath, header=None)
        # initial = zeros(nodes_num * K)
        initial = list(obs.iloc[0,:])


        network = network_estimation(ttime, dt, nodes, val_hidden, trails=2000, band_power=1. / float(node_dim + 1), K=K)
        solutions, time_line = network.simulation(val_hidden, nodes, initial, ttime, dt, 0, array([[]]), 2, net1=None,
                                                  net2=None,
                                                  true_net=False, hidden_network_fun=val_hidden)
        obs_t = time_line
        obs = solutions
        net_hidden = network.net2
        # hidden_network=network.network_func(evl,val_hidden)

        net1 = network.net1
        true_net = pd.DataFrame(evl, columns=['node1_1','node1_2','node1_3','node1_4','node1_5',
                                               'node2_1', 'node2_2', 'node2_3', 'node2_4', 'node2_5'])
        true_net['net_hidden'] = network.net2.flatten()

        # net_hidden = array(true_net['net_hidden']).reshape(nodes_num * K, nodes_num * K)

        desc = str(nodes_num) + 'x' + str(int(ttime / dt))
        rundate = t_module.strftime("%m%d%H%M", t_module.localtime())
        if filepath and os.path.exists(filepath):
            obs_filepath = self.save_path + 'obs_' + desc + ("_estimate_%d.csv" % test_num)
            true_net_filepath = self.save_path + 'true_net_' + desc + ("_estimate_%d.csv" % test_num)
        else:
            obs_filepath = self.save_path + 'obs_' + desc + '_original_' + rundate + '.csv'
            true_net_filepath = self.save_path + 'true_net_' + desc + '_original_' + rundate + '.csv'

        pd.DataFrame(obs).to_csv(obs_filepath, index=None, header=None)
        true_net.to_csv(true_net_filepath, index=None)
        print(obs_filepath)

        return obs_filepath, true_net_filepath


if __name__ == '__main__':
    rundate = t_module.strftime("%m%d%H%M", t_module.localtime())
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = BASE_DIR + '/../data/'

    filepath = save_path + 'to_file_true_net_11121656_re.csv'

    K = 2
    nodes_num = 100
    node_dim = 2
    time = 5
    dt = 0.05

    sim = simulation(save_path)
    obs_filepath, true_net_filepath = sim.do(K, nodes_num, node_dim, time, dt)




