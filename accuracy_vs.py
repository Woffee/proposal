"""
与不考虑分类的代码做对比
"""

import pandas as pd
import multiprocessing
import numpy as np

from numba import jit
import csv
import math
import datetime
import time
import scipy
import os
import math
from scipy.optimize import nnls
from scipy.stats import chi
import random
from clean_data import Clean_data
from simulation import simulation
from accuracy2 import accuracy
import logging
from accuracy2 import accuracy

def process(obs_filepath, new_filepath):
    data = pd.read_csv(obs_filepath, encoding='utf-8')
    data = np.array(data).T

    print(data.shape)
    res = []
    for i in range(len(data)):
        if i%2 ==0:
            # print(i)
            a = data[i]
            b = data[i+1]

            s = np.sum([a, b], axis=0)
            # print("sum of s:", np.sum(s))
            res.append(list(s))

    res = np.array(res).T
    res[res >= 1] = 1

    print(res.shape)
    pd.DataFrame(res).to_csv(new_filepath, index=None)

def get_accuracy(obs1_filepath, obs2_filepath):
    obs1 = pd.read_csv(obs1_filepath, encoding='utf-8')
    obs1 = np.array(obs1.values)
    sum1 = np.sum(obs1,axis=0) # axis=0 : column
    # print(sum1)

    obs2 = pd.read_csv(obs2_filepath, encoding='utf-8')
    obs2 = np.array(obs2.values)
    sum2 = np.sum(obs2, axis=0)  # axis=0 : column
    # print(sum2)

    s = 0
    for i in range(len(sum1)):
        if sum1[i] == sum2[i]:
            s = s + 1
    print("equal:", s, ",all:", len(sum1))
    return 1.0*s/len(sum1)


if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    rundate = time.strftime("%m%d%H%M", time.localtime())
    today = time.strftime("%Y-%m-%d", time.localtime())
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(filename)s line: %(lineno)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filename=BASE_DIR + '/log/' + today + '.log')

    save_path = BASE_DIR + '/data/vs2'

    # fp_obs = '/Users/woffee/www/proposal/data/vs2/obs_200x120_estimate_11200551.csv'
    # fp_obs_status = save_path + '/obs_status_original.csv'
    # fp_obs_status = save_path + '/obs_status_estimate.csv'
    # process(fp_obs, fp_obs_status)
    # exit(0)

    #
    # fp_obs = save_path + '/obs_150x120_original_11190318.csv'
    # fp_obs_status = save_path + '/obs_status_original.csv'
    # process(fp_obs, fp_obs_status)

    # obs_original_filepath = save_path + '/obs_status_original.csv'
    # ac_me = get_accuracy(obs_original_filepath, save_path + '/obs_status_estimate.csv')
    # print(ac_me)
    # ac_jx = get_accuracy(obs_original_filepath, save_path + '/obs_status_estimate_jx.csv')
    # print(ac_jx)

    # ac = accuracy(save_path + '/')
    # o1 = '/Users/woffee/www/proposal/data/vs/obs_150x120_original_11190318.csv'
    # o2 = '/Users/woffee/www/proposal/data/vs/obs_150x120_estimate_11190521.csv'
    # print(ac.get_accuracy1(o1, o2, 2, 150))


    oo = '/Users/woffee/www/proposal/data/vs/obs_status_original.csv'
    wb = '/Users/woffee/www/proposal/data/vs/obs_status_estimate.csv'
    jx = '/Users/woffee/www/network_research/data/vs/obs_status_estimate.csv'
    print("wb", get_accuracy(oo, wb))
    print("jx", get_accuracy(oo, jx))


