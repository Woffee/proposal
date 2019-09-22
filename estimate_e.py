# coding=utf-8
from sys import argv
from os.path import exists
import pandas as pd
import multiprocessing
from numpy import *
from numba import jit
import csv
import math
import datetime

from scipy import sparse as sp
from scipy.linalg import lstsq
from scipy.linalg import solve
from scipy.optimize import nnls
import scipy


@jit
def gaussiankernel(x, z, args, N):
    if N == 1:
        sigma = args
        y = (1. / sqrt(2. * pi) / sigma) * exp(-(x - z) ** 2 / (2. * sigma ** 2))
    else:
        sigma = args
        cov = []
        for j in range(N):
            cov += [1. / sigma[j, j] ** 2]
        N = float(N)

        y = 1. / (2. * pi) ** (N / 2.) * abs(linalg.det(sigma)) ** (-1.) * exp(
            (-1. / 2.) * dot((x - z) ** 2, array(cov)))
    return y



def construct_rxt(x):
    # construct r(x,t) in paper

    # x:
    # l.append(row)
    # l.append(ti)
    # l.append(features_matirx)
    # l.append(bandwidth)

    kernel = []
    n = x[2].shape[1]
    bandwidth = x[3]
    for row in x[2]:
        kernel.append(gaussiankernel(x[0], row, bandwidth, n))
    # print('kernel***********************************')
    # print(kernel)
    # (100,)   (200,)
    rxt_upper = dot(kernel, x[1])
    # print(t)
    # print(rxt_upper)
    rxt_lower = 0
    for i in kernel:
        rxt_lower = rxt_lower + i
    rxt = rxt_upper / rxt_lower
    return rxt


def minimizer_L1(x):
    D = x[1]
    y = -x[0].T
    # print('D>>>>>>>>>>>>>>>>>>>>>>')
    # print(D)
    # print('y>>>>>>>>>>>>>>>>>>>>>>')
    # print(y)

    edge_row, residual = nnls(D, y)
    # print(residual)
    # edge_row = MatchingPursuit(y,D,orthogonal=True)
    return edge_row



if __name__ == '__main__':
    # from_file = "/Users/woffee/www/network_research/data/example_data_for_fast_method/input.csv"
    from_file = "/Users/woffee/www/miningHiddenLink/proposal/data/input.csv"
    to_file   = "/Users/woffee/www/miningHiddenLink/proposal/data/to_file0918.csv"
    # script, from_file, to_file = argv
    # print(f"Reading from {from_file}  the result will be saved to {to_file}")
    # print(f"Does the output file exist? {exists(to_file)}")
    # print("Ready, hit RETURN to continue, CTRL-C to abort.")
    # input()

    start_time = datetime.datetime.now()
    # reading input data
    # data contains:feature vector, state
    data = pd.read_csv(from_file, encoding='utf-8')
    data.drop('user_id', axis=1, inplace=True)
    data = data.dropna()
    data.index = arange(len(data))

    ## get the features of nodes ##
    feature_sample = data[['FOLLOWERS_COUNT', 'FRIENDS_COUNT', 'STATUSES_COUNT', 'lat', 'lon']]
    feature_sample.index = data.index
    feature_col = feature_sample.columns
    ## rescale features to a compact cube ##
    feature_max = []
    for item in feature_sample.columns:
        feature_max.append(max(absolute(array(feature_sample[item]))))
        feature_sample[item] /= max(absolute(array(feature_sample[item])))
    ## define infect event and get 0-1 infection status sequence ##

    date_list = [20170811 + i for i in range(21)] + [20170901 + i for i in range(11)]
    K = 3
    spreading_key = []
    for k in range(K):
        for d, date in enumerate(date_list):
            spreading_key.append('k_' + str(k) + '_d_' + str(d))

    spreading_data = data[ spreading_key ]

    for k in range(K):
        prefix_key = 'k_' + str(k) + '_d_'
        for i in range(len(date_list)-1):
            spreading_data[prefix_key + str(i)] = spreading_data[prefix_key + str(i+1)] - spreading_data[prefix_key + str(i)]
        spreading_data[prefix_key + str(len(date_list)-1)] = 0
    spreading_sample = array(spreading_data)

    # spreading1 = ones_like(spreading_sample)
    # spreading1[spreading_sample < 1] = 0
    # spreading_sample = spreading1
    # nonz = where(spreading_sample[:, 0] != 0)[0]  # add back infection origins

    # draw subsample nodes and spreading info on these nodes
    # data_sample=random.choice(data.index,size=3)
    data_sample = range(0, 100)

    # for ind in nonz:
    #     if ind not in data_sample:
    #         data_sample = append(data_sample, array([ind], dtype=int))

    features = feature_sample.iloc[list(data_sample)]
    features.index = arange(len(data_sample))
    spreading = spreading_sample[list(data_sample)]



    # generate observation input
    obs = spreading.T
    # print('obs***********************\n')
    # print(obs.shape)
    features_matirx = features.values
    # print('features_onerow*************************')
    # print(features_matirx[0])
    bandwidth = diag(ones(features.shape[1]) * float(features.shape[0]) ** (-1. / float(features.shape[1] + 1)))
    # print('bandwidth*******************************')
    # print(bandwidth)

    # single_point_solver(features_matirx[0])
    rxt_params = []
    rxt_list = []

    for tix in obs:
        for row in features_matirx:
            l = []
            l.append(row)
            l.append(tix)
            l.append(features_matirx)
            l.append(bandwidth)
            rxt_params.append(l)
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    cnt = 0
    # construct r(x,t) in parallel
    # rxt_matrix is like:
    # r(x1,t1) .... r(xn,t1)
    # ...
    # r(x1,tm) .... r(xn,tm)
    rxt_list = pool.map(construct_rxt, rxt_params)
    rxt_matrix = asarray(rxt_list).reshape(obs.shape[0], features_matirx.shape[0])
    # construct all overline_r(x,t)
    overline_r_all = exp(1 - rxt_matrix.T)

    # compute differensiation
    # diff_x_all is like:
    # dx1t1 dx1t2 .... dx1tm
    # dx2t1 dx2t2 .... dx2tm
    # ...
    # dxnt1 dxnt2 .... dxntm
    overline_r_all2 = copy(overline_r_all)
    overline_r_all2_add = column_stack((overline_r_all2, overline_r_all2[:, -1]))
    overline_r_all2_splite = overline_r_all2_add[:, 1:]
    diff_x_all = overline_r_all2_splite - overline_r_all
    print("diff_x_all:", diff_x_all.shape)
    # reconstruct the compress sensing signal to get edge function
    xit_all = []
    for xit in diff_x_all:
        xit_matrix = []
        xit_matrix.append(xit)
        xit_matrix.append(rxt_matrix)
        xit_all.append(xit_matrix)
    edge_list = pool.map(minimizer_L1, xit_all)

    # print(xit_all[:20])
    print(edge_list[0])
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    with open(to_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(edge_list)

    # cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=cores)
    # cnt = 0
    # for y in pool.imap(single_point_solver, xs):
    #     sys.stdout.write('done %d/%d\r' % (cnt, len(xs)))
    #     cnt += 1
    # # deal with y
