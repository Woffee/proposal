# -*- coding: utf-8 -*-
"""
Created on Aug Sep 04 17:22:43 2019

@author: zxq
"""

from numpy import *
import time as t_module
import pandas as pd
from simulation_trick import network_estimation
from scipy.stats import chi2
from block import *
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_path = BASE_DIR + '/../data/simulation/'

filepath = save_path + 'true_net_1000x300.csv'

K=2
description = '2000x300'
nodes_num = 300 * K
node_dim = 2
time = 10
dt = 0.01
if os.path.exists(filepath) and False:
    file_net = pd.read_csv(filepath)
    nodes = file_net[['node1_x', 'node1_y']].iloc[range(nodes_num)].values
    val_hidden = file_net[description].values
else:
    # 1. generate nodes，这里的点就是用户向量，只不过是二维
    nodes = random.rand(nodes_num, node_dim)

    val_hidden = lambda x: 1 - sqrt(sum((x[:node_dim] - x[node_dim:]) ** 2)) / sqrt(2.)
    net_hidden1 = generate_network(nodes, val_hidden)

# 2. max possible edges counts, gererate those index
evl = []
for k in range(nodes.shape[0]):
    for j in range(nodes.shape[0]):
        evl.append(append(nodes[k], nodes[j]))
evl = array(evl)

hidden_net = []
initial = zeros(nodes_num)
for i in range(0, 50):  # 初始感染前50个节点
    initial[i] = 1

network = network_estimation(time, dt, nodes, val_hidden, trails=2000, band_power=1. / float(node_dim + 1), K=K)
solutions, time_line = network.simulation(val_hidden, nodes, initial, time, dt, 0, array([[]]), 2, net1=None, net2=None,
                                          true_net=False, hidden_network_fun=val_hidden)
obs_t = time_line
obs = solutions
net_hidden = network.net2
# hidden_network=network.network_func(evl,val_hidden)

net1 = network.net1
true_net = pd.DataFrame(evl, columns=['node0_x', 'node0_y', 'node1_x', 'node1_y'])
true_net['net_hidden'] = network.net2.flatten()

net_hidden = array(true_net['net_hidden']).reshape(nodes_num, nodes_num)

if os.path.exists(filepath) and False:
    # pd.DataFrame(obs_t).to_csv(save_path + 'obst_1000x300_' + description + '_re' + '.csv')  # 这个没用了
    pd.DataFrame(obs).to_csv(save_path + 'obs_000x300_' + description + '_re' + '.csv')
    true_net.to_csv(save_path + 'true_net_1000x300_' + description + '_re' + '.csv')
else:
    # pd.DataFrame(obs_t).to_csv(save_path + 'obst_' + description + '.csv')
    pd.DataFrame(obs).to_csv(save_path + 'obs_' + description + '.csv')
    true_net.to_csv(save_path + 'true_net_' + description + '.csv')
