# coding=utf-8
import pandas as pd
import multiprocessing
# import numpy as np
from numpy import *
from numba import jit
import csv

import datetime
import time
# import cvxpy as cvx
# from scipy import sparse as sp
# from scipy.linalg import lstsq
# from scipy.linalg import solve
from scipy.optimize import nnls
import scipy
import os


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


def square_sum(x):
    y = dot(x,x)
    return y

# def lessObsLwConstrain(x,D,y):
#     temp = y - dot(D,x.reshape(len(x),1))
#     temp = temp.reshape(1,len(temp))
#     eq = asscalar(dot(temp,temp.T))
#     return eq-0.001

def lasso(x,D,y):
    temp = (dot(D,x))/D.shape[1] - y
    eq = dot(temp,temp)
    return eq+dot(x,x)

def lessObsUpConstrain(x,D,y):
    temp = (dot(D,x))/D.shape[1] - y
    eq = dot(temp,temp)
    return -eq+0.1

def moreObsfunc(x,D,y):
    temp = y.reshape(len(y),1)-dot(D,x.reshape(len(x),1))
    temp = temp.reshape(1,len(temp))
    return asscalar(dot(temp,temp.T))

def minimizer_L1(x):
    D=x[1]
    y=x[0].T
    print('D>>>>>>>>>>>>>>>>>>>>>>')
    print(D)
    print('y>>>>>>>>>>>>>>>>>>>>>>')
    print(y)
    # x0=x[2].reshape(D.shape[1],)-(random.rand(D.shape[1]))/100
    x0=ones(D.shape[1],)
    # print('guess x0>>>>>>>>>>')
    # print(x0)
    print("D:", D.shape)
    if(D.shape[0] < D.shape[1]):
        #less observations than nodes
        #edge_row = MatchingPursuit(y,D,orthogonal=True)
        upcons = {'type':'ineq','fun':lessObsUpConstrain,'args':(D,y)}
        result = scipy.optimize.minimize(square_sum, x0, args=(), method='SLSQP', jac=None, bounds=scipy.optimize.Bounds(0,1), constraints=[upcons], tol=None, callback=None, options={'maxiter': 100, 'ftol': 1e-06, 'iprint': 1, 'disp': False, 'eps': 1.4901161193847656e-08})
        #result = scipy.optimize.minimize(lasso, x0, args=(D,y), method='L-BFGS-B', jac=None, bounds=scipy.optimize.Bounds(0,1), tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
        print(result)
        #edge_row = NonnegativeBP(y,D)#can get non negative tiny value result
    else:
        #observations is more than nodes
        #edge_row = linalg.lstsq(D,y)
        #edge_row,residual = nnls(D,y,maxiter=None)
        result = scipy.optimize.minimize(moreObsfunc, x0, args=(D,y), method='L-BFGS-B', jac=None, bounds=scipy.optimize.Bounds(0,1), tol=None, callback=None, options={'disp': None, 'maxcor': 10, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': -1, 'maxls': 20})
        print(result)
    return result.x

def meanfield(mf):
    E = mf[0]
    n = mf[1]
    res = dot(E,n)
    # print(res)
    return res

def get_edge_list(rxt_matrix, pool):
    # print(rxt_matrix.shape)
    # (32, 120)

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
    # diff_x_all: (120, 32)


    # zero_row_ids = []
    # nonezero_row_ids = []
    # row_id_in_diff_x_all = 0
    # for row in diff_x_all:
    #     if (all(row == 0)):
    #         zero_row_ids.append(row_id_in_diff_x_all)
    #     else:
    #         nonezero_row_ids.append(row_id_in_diff_x_all)
    #     row_id_in_diff_x_all = row_id_in_diff_x_all + 1
    #
    # diff_x_nonezero = delete(diff_x_all, zero_row_ids, 0)
    # for i in range(len(nonezero_row_ids) - 1):
    #     diff_x_nonezero[i, :] = diff_x_nonezero[i, :] / ((nonezero_row_ids[i + 1] - nonezero_row_ids[i]) * dt)
    #
    # D = delete(rxt_matrix, zero_row_ids, 0)

    # reconstruct the compress sensing signal to get edge function
    xit_all = []
    for xit in diff_x_all:
        xit_matrix = []
        xit_matrix.append(xit)  # y
        # print("xit:",len(xit))
        # print("rxt_matrix:",rxt_matrix.shape)
        # xit: 32
        # rxt_matrix: (32, 120)

        # xit_matrix.append(D)
        xit_matrix.append(rxt_matrix) # D
        xit_all.append(xit_matrix)
    edge_list = pool.map(minimizer_L1, xit_all)
    return edge_list

if __name__ == '__main__':

    # tt = asarray( [ [1,2,3,4,5,6,7,8,9],
    #        [11,22,33,44,55,66,77,88,99]])
    # print(tt.shape)
    # tt = tt.reshape((6,3))
    # print(tt)
    # exit(0)

    dt = 0.1

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = BASE_DIR + '/data/'

    from_file = save_path + "input.csv"
    rundate = time.strftime("%m%d%H%M", time.localtime())
    to_file   = save_path + "to_file" + rundate + ".csv"
    to_file2 = save_path + "to_file_2_" + rundate + ".csv"
    print(to_file)


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
    K = 2
    spreading_key = []
    for k in range(K):
        for d, date in enumerate(date_list):
            spreading_key.append('k_' + str(k) + '_d_' + str(d))

    spreading_data = data[ spreading_key ]

    # for k in range(K):
    #     prefix_key = 'k_' + str(k) + '_d_'
    #     for i in range(len(date_list)-1):
    #         spreading_data[prefix_key + str(i)] = spreading_data[prefix_key + str(i+1)] - spreading_data[prefix_key + str(i)]
    #     spreading_data[prefix_key + str(len(date_list)-1)] = 0
    spreading_sample = array(spreading_data)

    sample_size = 60
    data_sample = range(0, sample_size)
    # spreading_sample = spreading_sample.reshape( (sample_size * K, len(date_list)) )

    # spreading1 = ones_like(spreading_sample)
    # spreading1[spreading_sample < 1] = 0
    # spreading_sample = spreading1
    nonz = where(spreading_sample[:, 0] != 0)[0]  # add back infection origins

    # draw subsample nodes and spreading info on these nodes
    # data_sample=random.choice(data.index,size=3)


    for ind in nonz:
        if ind not in data_sample:
            data_sample = append(data_sample, array([ind], dtype=int))

    features = feature_sample.iloc[list(data_sample)]
    features.index = arange(len(data_sample))
    spreading = spreading_sample[list(data_sample)]
    spreading = spreading.reshape( (sample_size * K, len(date_list)) )



    # generate observation input
    obs = spreading.T
    # print('obs***********************\n')
    # print(obs.shape)

    # features_matirx = features.values
    features_matirx = []
    for v in features.values:
        for i in range(K):
            features_matirx.append(v)
    features_matirx = array(features_matirx)


    # print('features_onerow*************************')
    # print(features_matirx[0])
    bandwidth = diag(ones(features.shape[1]) * float(features.shape[0]) ** (-1. / float(features.shape[1] + 1)))
    # print('bandwidth*******************************')
    # print(bandwidth)

    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)

    # single_point_solver(features_matirx[0])
    rxt_params = []
    for tix in obs:
        for row in features_matirx:
            l = []
            l.append(row)
            l.append(tix)
            l.append(features_matirx)
            l.append(bandwidth)
            rxt_params.append(l)
    # cnt = 0
    # construct r(x,t) in parallel
    # rxt_matrix is like:
    # r(x1,t1) .... r(xn,t1)
    # ...
    # r(x1,tm) .... r(xn,tm)
    rxt_list = pool.map(construct_rxt, rxt_params)
    rxt_matrix = asarray(rxt_list).reshape(obs.shape[0], features_matirx.shape[0])

    np.savetxt( save_path + 'to_file_rxt_matrix_' + rundate + ".txt", rxt_matrix)


    edge_list = get_edge_list(rxt_matrix, pool)

    meanfield_list = []
    for tix in obs:
        item = []
        item.append(edge_list)
        item.append(tix)
        meanfield_list.append(item)
    new_rxt_matrix = pool.map(meanfield, meanfield_list)

    # print(xit_all[:20])
    print(edge_list[0])
    end_time = datetime.datetime.now()
    print(end_time - start_time)
    with open(to_file, "w") as f:
        writer = csv.writer(f)
        writer.writerows(edge_list)
    print(to_file)

    new_edge_list = get_edge_list(asarray(new_rxt_matrix), pool)
    with open(to_file2, "w") as f:
        writer = csv.writer(f)
        writer.writerows(new_edge_list)
    print(to_file2)

    # meanfield_list = []
    # for tix in obs:
    #     item = []
    #     item.append(new_edge_list)
    #     item.append(tix)
    #     meanfield_list.append(item)
    # new_rxt_matrix = pool.map(meanfield, meanfield_list)
    #
    # new_edge_list = get_edge_list(asarray(new_rxt_matrix), pool)
    # to_file2 = save_path + "to_file_3_" + time.strftime("%m%d", time.localtime()) + ".csv"
    # with open(to_file2, "w") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(new_edge_list)
    # print(to_file2)

    # cores = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=cores)
    # cnt = 0
    # for y in pool.imap(single_point_solver, xs):
    #     sys.stdout.write('done %d/%d\r' % (cnt, len(xs)))
    #     cnt += 1
    # # deal with y
