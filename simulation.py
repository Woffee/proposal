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

    def do(self, K, nodes_num, node_dim, time, dt, filepath = None):
        if filepath and os.path.exists(filepath):
            file_net = pd.read_csv(filepath)
            nodes = file_net[['node1_x', 'node1_y']].iloc[range(nodes_num * K)].values
            val_hidden = file_net['e'].values
        else:
            # 1. generate nodes
            nodes = []
            rnd_nodes = random.rand(nodes_num, node_dim)
            for node in rnd_nodes:
                for k in range(K):
                    nodes.append(node)
            nodes = array(nodes)

            val_hidden = lambda x: 1 - sqrt(sum((x[:node_dim] - x[node_dim:]) ** 2)) / sqrt(2.)
            net_hidden1 = generate_network(nodes, val_hidden)

        # 2. max possible edges counts, gererate those index
        evl = []
        for k in range(nodes.shape[0]):
            for j in range(nodes.shape[0]):
                evl.append(append(nodes[k], nodes[j]))
        evl = array(evl)

        hidden_net = []
        initial = zeros(nodes_num * K)
        tmp = int(nodes_num * K / 4)
        for i in range(0, tmp):  # 初始感染节点
            initial[i * 4] = 1

        network = network_estimation(time, dt, nodes, val_hidden, trails=2000, band_power=1. / float(node_dim + 1), K=K)
        solutions, time_line = network.simulation(val_hidden, nodes, initial, time, dt, 0, array([[]]), 2, net1=None,
                                                  net2=None,
                                                  true_net=False, hidden_network_fun=val_hidden)
        obs_t = time_line
        obs = solutions
        net_hidden = network.net2
        # hidden_network=network.network_func(evl,val_hidden)

        net1 = network.net1
        true_net = pd.DataFrame(evl, columns=['node0_x', 'node0_y', 'node1_x', 'node1_y'])
        true_net['net_hidden'] = network.net2.flatten()

        # net_hidden = array(true_net['net_hidden']).reshape(nodes_num * K, nodes_num * K)

        desc = str(nodes_num) + 'x' + str(int(time / dt))
        rundate = t_module.strftime("%m%d%H%M", t_module.localtime())
        if filepath and os.path.exists(filepath):
            obs_filepath = self.save_path + 'obs_' + desc + '_estimate_' + rundate + '.csv'
            true_net_filepath = self.save_path + 'true_net_' + desc + '_estimate_' + rundate + '.csv'
        else:
            obs_filepath = self.save_path + 'obs_' + desc + '_original_' + rundate + '.csv'
            true_net_filepath = self.save_path + 'true_net_' + desc + '_original_' + rundate + '.csv'

        pd.DataFrame(obs).to_csv(obs_filepath, index=None)
        true_net.to_csv(true_net_filepath, index=None)
        print(obs_filepath)
        print(true_net_filepath)

        return obs_filepath, true_net_filepath




