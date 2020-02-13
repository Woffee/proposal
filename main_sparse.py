"""
Version 3，暂时不考虑迭代，假设分类已知
验证稀疏矩阵
"""

import pandas as pd
import time
import os
from simulation import simulation
from accuracy2 import accuracy
import logging
from main import MiningHiddenLink

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

    test = [
        [130, 120],
        [140, 120],
        [150, 120],
        [160, 120],
        [170, 120],
        [180, 120],
        [190, 120],
        [200, 120],
    ]
    for nodes_num, obs_num in test:
        time = 1.0 * obs_num * dt
        print(nodes_num, obs_num, time)
        logging.info("sparse vali start: " + str(nodes_num) + "x" + str(obs_num))

        save_path = BASE_DIR + '/data/' + str(nodes_num) + "x" + str(obs_num)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        files = os.listdir(save_path)
        e_filepath = ''
        true_net_filepath = ''
        obs_filepath = ''

        for f in files:
            if 'to_file_E' in f:
                e_filepath = save_path + "/" + str(f)
                print(e_filepath)
            if 'to_file_true_net_' in f:
                true_net_filepath = save_path + "/" + str(f)
                print(true_net_filepath)
            if 'obs_' in f and 'original' in f:
                obs_filepath = save_path + "/" + str(f)
                print(obs_filepath)


        if e_filepath == '' or true_net_filepath =='':
            logging.info("not found !")
            print("not found !")
            exit(0)

        save_path = save_path + "/"

        sim = simulation(save_path)
        mhl = MiningHiddenLink(save_path)
        ac = accuracy(save_path)

        true_net_re_filepath = save_path + "to_file_true_net_sparse_" + rundate + "_re.csv"
        true_net = pd.read_csv(true_net_filepath, sep=',')
        hidden_link = pd.read_csv(e_filepath, sep=',', header=None)
        hidden_link[ hidden_link <= 0.5 ] = 0
        hidden_link[ hidden_link > 0.5 ] = 1
        true_net['e'] = hidden_link.values.flatten()
        true_net.to_csv(true_net_re_filepath, header=True, index=None)
        logging.info("step 1: " + true_net_re_filepath)

        obs_filepath_2, true_net_filepath_2 = sim.do(K, nodes_num, node_dim, time, dt, true_net_re_filepath)
        logging.info("step 2: " + obs_filepath_2)
        logging.info("step 2: " + true_net_filepath_2)

        # 5
        a1 = ac.get_accuracy1(obs_filepath, obs_filepath_2, K, nodes_num)
        print("accuracy1:", a1)
        a2 = ac.get_accuracy2(obs_filepath, obs_filepath_2, true_net_filepath, true_net_filepath_2, K, nodes_num)
        print("accuracy2:", a2)
        logging.info("step 3 " + str(nodes_num) + "x" + str(obs_num) + " accuracy1: " + str(a1))
        logging.info("step 3 " + str(nodes_num) + "x" + str(obs_num) + " accuracy2: " + str(a2))

        logging.info("done")

