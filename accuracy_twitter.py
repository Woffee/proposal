import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = BASE_DIR + '/data_twitter'

class accuracy:
    def __init__(self, save_path):
        self.save_path = save_path

    def distance(self, a,b):
        s = 0
        for i in range(len(a)):
            s = s + (a[i] - b[i])**2
        return s

    def find_near(self, obj, all):
        dist = self.distance(obj,all[0])
        near = all[0]
        for line in all:
            d = self.distance(obj,line)
            if d < dist:
                dist = d
                near = line
        return near[0], near[1], near[2], near[3], near[4], near[5], near[6], near[7]

    def cal_accuracy(self, x, x_near):
        near_dot = np.dot(x_near,x_near)
        absdiff = x - x_near
        return (near_dot / (np.dot(absdiff,absdiff) + near_dot))

    def process_data(self, filepath_original, filepath_new, filepath_true_net):
        filepath_result = self.save_path + 'true_net_100_result.csv'
        if os.path.exists(filepath_result):
            return filepath_result

        obs_original = pd.read_csv(filepath_original, encoding='utf-8').dropna()
        obs_new = pd.read_csv(filepath_new, encoding='utf-8').dropna()
        true_net = pd.read_csv(filepath_true_net, encoding='utf-8').dropna()

        sum_1 = np.sum(obs_original.values, axis=0)
        sum_2 = np.sum(obs_new.values, axis=0)

        data_sample = range(0, len(sum_1))
        result = true_net[['node1_x', 'node1_y']]
        result = result.iloc[list(data_sample)]

        result['original'] = sum_1
        result['original'] = result['original'] / 1000

        result['time5'] = sum_2
        result['time5'] = result['time5'] / 1000

        pd.DataFrame(result).to_csv(filepath_result, index=None)
        print(filepath_result)
        return filepath_result

    def recovery_process(self, original):
        for j in range(original.shape[1]):
            # 超过3天不变，改为0
            last = -1
            s = 1
            for i in range(original.shape[0]):
                if original[i][j] == last:
                    s = s + 1
                else:
                    s = 1
                    last = original[i][j]
                if s > 3:
                    original[i][j] = 0
                # print(original[i][j])

            # 把数字改成从1开始
            i = 0
            last = 0
            while(i < original.shape[0]):
                now = original[i][j]
                # print(type(now))
                k = i
                if (now - last) > 1:
                    while(k+1 < original.shape[0] and original[k+1][j] == now):
                        k = k+1
                    for l in range(i, k+1):
                        original[l][j] = last + 1
                last = original[k][j]
                i = k + 1
            # for i in range(original.shape[0]):
            #     print(original[i][j])
            # exit()
        return original

    # success rate
    def get_accuracy1(self, filepath_o, filepath_e, K, num_nodes):
        original = pd.read_csv(filepath_o, encoding='utf-8', header=None).dropna().to_numpy()
        estimate = pd.read_csv(filepath_e, encoding='utf-8', header=None).dropna().to_numpy()

        original = self.recovery_process(original)
        estimate = self.recovery_process(estimate)

        np.savetxt(SAVE_PATH + '/obs_rec.csv', original, delimiter=',', fmt='%d')
        np.savetxt(SAVE_PATH + '/obs_estimate_rec.csv', estimate, delimiter=',', fmt='%d')

        if original.shape[0] < estimate.shape[0]:
            estimate = np.delete(estimate, -1, axis = 0)
        # print(original.shape)
        # print(estimate.shape)
        s = 0
        for i in range(num_nodes):
            equal = True
            for j in range(K):
                ii = i*K + j
                col1 = original[:, ii]
                col2 = estimate[:, ii]
                if not (col1 == col2).all():
                    equal = False
                    break
            if equal:
                # print("equal:", i)
                s = s + 1
        print("equal:", s, ",all:",num_nodes)
        return 1.0*s/num_nodes

    def get_data(self, fp_obs, fp_true_net, K, num_nodes):
        true_net = pd.read_csv(fp_true_net, encoding='utf-8').dropna()
        obs      = pd.read_csv(fp_obs, encoding='utf-8', header=None).dropna().to_numpy()
        obs      = self.recovery_process(obs)

        T = len(obs)
        nodes = true_net[['node1_1','node1_2','node1_3','node1_4','node1_5']].drop_duplicates().to_numpy()

        data = []
        for i in range(num_nodes):
            node = nodes[i]
            for t in range(T):
                row = [node[0], node[1], node[2], node[3], node[4]]
                for k in range(K):
                    row.append( obs[t, i*K + k] )
                row.append(t)
                # print(row)
                data.append(row)
        return np.array(data)

    def get_accuracy2(self, fp_obs_o, fp_obs_e, fp_true_net_o, fp_true_net_e, K, num_nodes):
        data_o = self.get_data(fp_obs_o, fp_true_net_o, K, num_nodes)
        data_e = self.get_data(fp_obs_e, fp_true_net_e, K, num_nodes)

        # data = pd.read_csv(fp_true_net_o, encoding='utf-8').dropna()
        # nodes = data[['node1_x', 'node1_y']]
        # nodes = nodes.drop_duplicates()

        x1_near = []
        x2_near = []
        x3_near = []
        x4_near = []
        x5_near = []

        n1_near = []
        n2_near = []
        t_near = []

        for i in range(len(data_o)):
            # print(i)
            if (data_o[i] == data_e[i]).all():
                x1_near.append(data_o[i, 0])
                x2_near.append(data_o[i, 1])
                x3_near.append(data_o[i, 2])
                x4_near.append(data_o[i, 3])
                x5_near.append(data_o[i, 4])

                n1_near.append(data_o[i, 5])
                n2_near.append(data_o[i, 6])
                t_near.append(data_o[i, 7])
            else:
                x1near, x2near, x3near, x4near, x5near, n1near, n2near, tnear = self.find_near(data_e[i], data_o)
                x1_near.append(x1near)
                x2_near.append(x2near)
                x3_near.append(x3near)
                x4_near.append(x4near)
                x5_near.append(x5near)

                n1_near.append(n1near)
                n2_near.append(n2near)
                t_near.append(tnear)

        # x = nodes[['node1_x']].to_numpy().T.reshape(num_nodes * K, )
        # y = nodes[['node1_y']].to_numpy().T.reshape(num_nodes * K, )

        x1  = data_e[:,0]
        x2  = data_e[:,1]
        x3  = data_e[:,2]
        x4  = data_e[:,3]
        x5  = data_e[:,4]

        n1 = data_e[:,5]
        n2 = data_e[:,6]
        t  = data_e[:,7]

        x1_near = np.asarray(x1_near)
        x2_near = np.asarray(x2_near)
        x3_near = np.asarray(x3_near)
        x4_near = np.asarray(x4_near)
        x5_near = np.asarray(x5_near)

        n1_near = np.asarray(n1_near)
        n2_near = np.asarray(n2_near)
        t_near = np.asarray(t_near)

        accuracy_x1  = self.cal_accuracy(x1, x1_near)
        accuracy_x2  = self.cal_accuracy(x2, x2_near)
        accuracy_x3  = self.cal_accuracy(x3, x3_near)
        accuracy_x4  = self.cal_accuracy(x4, x4_near)
        accuracy_x5  = self.cal_accuracy(x5, x5_near)

        accuracy_n1 = self.cal_accuracy(n1, n1_near)
        accuracy_n2 = self.cal_accuracy(n2, n2_near)
        accuracy_t  = self.cal_accuracy(t, t_near)

        print(accuracy_x1)
        print(accuracy_x2)
        print(accuracy_x3)
        print(accuracy_x4)
        print(accuracy_x5)

        print(accuracy_n1)
        print(accuracy_n2)
        print(accuracy_t)

        accuracy = (accuracy_x1 + accuracy_x2 + accuracy_x3 + accuracy_x4 + accuracy_x5 + accuracy_n1 + accuracy_n2 + accuracy_t) / 8
        return accuracy




if __name__ == '__main__':
    K = 2
    num_nodes = 100
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = BASE_DIR + '/data/'

    fp_obs_o = save_path+'obs_100x100_original.csv'
    fp_obs_e = save_path+'obs_100x100_estimate_11121658.csv'

    fp_true_net_o = save_path+'true_net_100x100_original.csv'
    fp_treu_net_e = save_path+'true_net_100x100_estimate_11121658.csv'

    ac = accuracy(save_path)

    a1 = ac.get_accuracy1(fp_obs_o, fp_obs_e, K, num_nodes)
    a2 = ac.get_accuracy2(fp_obs_o, fp_obs_e, fp_true_net_o, fp_treu_net_e, K, num_nodes)
    print("accuracy1:", a1)
    print("accuracy2:", a2)






