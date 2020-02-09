"""
权重  < 0.5 赋值为 0
权重 >= 0.5 赋值为 1

用上面的方式，处理 真实E 和 估计E。

然后对比两个矩阵 entry-wise 之差，用这个差矩阵刻画二者的差别。

By Wenbo
Feb 5, 2020
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
import numpy as np
import math
import time
import scipy
import os
import logging


BINS = 40

def get_percentage(original):
    hist, bins = np.histogram(original[:, 1], bins=20)
    # print("hist:", hist)
    # print("bins:", bins)
    sum = len(original)
    perc = list(map(lambda x: 1.0 * x / sum, hist))
    # print("perc", perc)

    original_size = []
    for o in original:
        for i in range(20):
            if o[1] < bins[i + 1]:
                original_size.append(perc[i])
    return original_size

def show_data(data):
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    # ax1.ticklabel_format(useOffset=False)

    # see https://matplotlib.org/tutorials/intermediate/legend_guide.html
    red_patch = patches.Patch(color='tab:red', label='Estimate')
    blue_patch = patches.Patch(color='tab:blue', label='Original')
    plt.legend(handles=[red_patch, blue_patch], loc='upper right')
    # plt.legend(handles=[l1,l2], labels=['original','estimate'],loc='upper right')

    h = 1.5 / BINS
    for index, row in data.iterrows():
        x = float(row['nodes'])
        y = float(row['dist'])
        w = float(row['perc']) * 40
        c = 'tab:blue' if row['type'] == 'original' else 'tab:red'
        # print(x,y,w,h)

        ax1.add_patch(
            patches.Rectangle(
                (x, y),  # (x,y)
                w,  # width
                h,  # height
                color=c,
                alpha=0.5,
                ec = None
            )
        )
    plt.xticks([150, 160, 170, 180, 190, 200])

    plt.xlim(155, 205)
    plt.ylim(-0.05, 1.5)

    plt.xlabel("Number of nodes")
    plt.ylabel("Distance of each node pair")

    # plt.yticks([0,0.5,1])
    plt.show()



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
        # [130, 100],
        # [140, 100],
        # [150, 100],
        [160, 100],
        [170, 100],
        [180, 100],
        [190, 100],
        [200, 100],
    ]

    data_toshow_file = './data_to_show.csv'
    if not os.path.exists(data_toshow_file):

        # fig, axs = plt.subplots(len(test), 1, sharey=True, tight_layout=True)
        ii = 0

        data_toshow_nodesnum = []
        data_toshow_dist = []
        data_toshow_perc = []
        data_toshow_type = []

        for nodes_num, obs_num in test:
            original = []
            estimate = []
            # time = 1.0 * obs_num * dt
            print(nodes_num, obs_num)
            # logging.info("sparse vali start: " + str(nodes_num) + "x" + str(obs_num))

            save_path = BASE_DIR + '/data/' + str(nodes_num) + "x" + str(obs_num)
            if not os.path.exists(save_path):
                print("not exists:",save_path)
                exit()

            files = os.listdir(save_path)
            e_filepath = ''
            true_net_filepath = ''
            obs_filepath = ''

            for f in files:
                if 'to_file_true_net_' in f and not 'sparse' in f:
                    true_net_filepath = save_path + "/" + str(f)
                    print(true_net_filepath)
                    break

            save_path = save_path + "/"

            data = pd.read_csv(true_net_filepath, sep=',')

            # df['col3'] = df.apply(lambda x: x['col1'] + 2 * x['col2'], axis=1)
            # locations = true_net[['node0_x','node0_y','node1_x','node1_y']]

            data['dist'] = data.apply( lambda x: math.sqrt( (x['node0_x'] - x['node1_x'] )**2 + (x['node0_y'] - x['node1_y'])**2  ), axis=1)
            print(data.head())

            true_net = data[['net_hidden','e']]
            true_net[true_net<0.5] = 0
            true_net[true_net>=0.5] = 1


            for index, row in true_net.iterrows():
                # print(row['net_hidden'], row['e'])
                if row['net_hidden'] > 0:
                    original.append([nodes_num-1, data.iloc[index, 6]])
                if row['e'] > 0:
                    estimate.append([nodes_num+1, data.iloc[index, 6]])


            # original_e = true_net.net_hidden.to_numpy()
            # estimate_e = true_net.e.to_numpy()
            #
            # equal = sum(original_e == estimate_e)
            # all = len(original_e)
            #
            # log_info = str(nodes_num) + "x" + str(obs_num) + ", equal: %d, all: %d, same rate:%.6f" % ( equal, all, 1.0*equal/all)
            # print(log_info)
            # logging.info(log_info)

            original = np.array(original)
            estimate = np.array(estimate)

            len_o = len(original)
            print("len(original)", len(original))
            print("len(estimate)", len(estimate))

            hist, bins = np.histogram(original[:, 1], bins=BINS, range=(0, 1.5))
            print("bins:", bins)
            hist = list(map(lambda x: 1.0*x/len_o, hist))
            for i,h in enumerate(hist):
                data_toshow_nodesnum.append(nodes_num)
                data_toshow_dist.append(bins[i])
                data_toshow_perc.append(hist[i])
                data_toshow_type.append('original')

            hist, bins = np.histogram(estimate[:, 1], bins=BINS, range=(0, 1.5))
            hist = list(map(lambda x: 1.0 * x / len_o, hist))
            for i, h in enumerate(hist):
                data_toshow_nodesnum.append(nodes_num)
                data_toshow_dist.append(bins[i])
                data_toshow_perc.append(hist[i])
                data_toshow_type.append('estimate')


        data = pd.DataFrame({'nodes':data_toshow_nodesnum,'dist':data_toshow_dist,'perc':data_toshow_perc,'type':data_toshow_type})
        data.to_csv(data_toshow_file, index=False)
    else:
        data = pd.read_csv(data_toshow_file)

    show_data(data)


    # np.savetxt('./original.txt', original, fmt='%.8f')
    # np.savetxt('./estimate.txt', estimate, fmt='%.8f')
    # exit(0)



    # original = np.loadtxt('./original.txt')
    # estimate = np.loadtxt('./estimate.txt')

    # perc_o = get_percentage(original)
    # perc_e = get_percentage(estimate)





    # size_o = list( map(lambda x: x*2, perc_o ) )
    # size_e = list( map(lambda x: x*2, perc_e ) )
    #
    # # plt.figure(figsize=(10, 5))
    # # plt.title("Success Rate when Choosing Different Time Steps")
    # plt.xlabel("Number of nodes")
    # # plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    # plt.ylabel("Distance of each node pair")
    # # plt.ylim(0.95, 1)
    #
    #
    # plt.scatter(original[:, 0], original[:, 1], c='tab:blue', s=size_o, marker='o', alpha=0.3)
    # plt.scatter(estimate[:, 0], estimate[:, 1], c='tab:red',  s=size_e, marker='o', alpha=0.3)
