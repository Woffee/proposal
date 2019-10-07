"""

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



def is_constrained(E1, E2):
    return True


def constrained_clustering(C):
    return C


@jit
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

def read_data(filepath, K, days, sample_size = 60):
    # reading input data
    # data contains:feature vector, state
    data = pd.read_csv(filepath, encoding='utf-8')
    data.drop('user_id', axis=1, inplace=True)
    data = data.dropna()
    data.index = np.arange(len(data))

    ## get the features of nodes ##
    feature_sample = data[['FOLLOWERS_COUNT', 'FRIENDS_COUNT', 'STATUSES_COUNT', 'lat', 'lon']]
    feature_sample.index = data.index
    # feature_col = feature_sample.columns
    ## rescale features to a compact cube ##
    # feature_max = []
    feature_sample['lon'] += 180
    # print(feature_sample[['lat', 'lon']][:5])
    for item in feature_sample.columns:
        # feature_max.append(max(np.absolute(np.array(feature_sample[item]))))
        feature_sample[item] /= max(np.absolute(np.array(feature_sample[item])))


    ## define infect event and get 0-1 infection status sequence ##

    spreading_key1 = []
    spreading_key2 = []
    for k in range(K):
        for d in range(days):
            if d%2==0:
                spreading_key1.append('k_' + str(k) + '_d_' + str(d))
            else:
                spreading_key2.append('k_' + str(k) + '_d_' + str(d))

    spreading_sample1 = np.array(data[spreading_key1])
    # spreading_sample1 = spreading_sample1.reshape((sample_size * K / 2, days))
    spreading_sample2 = np.array(data[spreading_key2])
    # spreading_sample2 = spreading_sample2.reshape((sample_size * K / 2, days))

    sample_range = range(0, sample_size)
    features = feature_sample.iloc[list(sample_range)]
    features.index = np.arange(len(sample_range))
    spreading1 = spreading_sample1[list(sample_range)].reshape((sample_size * K, int(days/K)))
    spreading2 = spreading_sample2[list(sample_range)].reshape((sample_size * K, int(days/K)))

    return [features,spreading1], [features, spreading2]

def get_n_matrix(sample):
    return []

# 1.1 & 1.3
def get_r_xit(x, i, t_l, N, n_maxtrix):
    numerator = 0.0
    denominator = 0.0

    for j in range(N):
        g = gaussiankernel() # todo
        tmp = n_maxtrix[j*i][t_l+1] - n_maxtrix[j*i][t_l]
        numerator = numerator + (g*tmp)
        denominator = denominator + (g*2)
    return numerator/denominator

def get_r_matrix(n_matrix):
    return []

# 1.1 & 1.4
def get_E(rxt_matrix, n_matrix):
    return []

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = BASE_DIR + '/data/'

    K = 2
    days = 32
    sample_size = 60
    sample1, sample2 = read_data(save_path+'input.csv', K, days, sample_size)

    print(sample1[0][:5])
    print(sample1[1][:50])

    rxt_matrix1 = get_r_matrix(sample1)
    n_matrix1 = get_n_matrix(sample1)
    E1 = get_E(rxt_matrix1, n_matrix1)

    rxt_matrix2 = get_r_matrix(sample2)
    n_matrix2 = get_n_matrix(sample2)
    E2 = get_E(rxt_matrix2, n_matrix2)




    # 1. given classified texts C
    # and two sets of observation samples at different times(T1,T2)
    C=[]

    while (True):
        # 2. calculate E1 and E2
        E1 = []
        E2 = []

        # 3. see if constrained
        constrained = is_constrained(E1, E2)
        if constrained:
            break

        # 4. update text classifications
        C = constrained_clustering(C)

    # 5. output results