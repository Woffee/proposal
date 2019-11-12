import pandas as pd
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_path = BASE_DIR + '/data/'

def distance(a,b):
    return (a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2

def find_near(obj, all):
    dist = distance(obj,all[0])
    near = []
    for line in all:
        d = distance(obj,line)
        if d < dist:
            dist = d
            near = line
    return near[0], near[1], near[2]

def cal_accuracy(x, x_near):
    near_dot = np.dot(x_near,x_near)
    absdiff = x - x_near
    return (near_dot / (np.dot(absdiff,absdiff) + near_dot))

def process_data(filepath_original, filepath_new, filepath_true_net):
    filepath_result = save_path + 'true_net_100_result.csv'
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


def get_accuracy1(from_file, K, num_nodes, test_name):
    data = pd.read_csv(from_file, encoding='utf-8').dropna()
    test_nodes = data[['original', test_name]]
    test_matrix = test_nodes.to_numpy()

    s = 0
    for row in test_matrix:
        if row[0] == row[1]:
            s = s+1
    return 1.0*s/(len(test_matrix))

def get_accuracy2(from_file, K, num_nodes, test_name):
    # reading input data
    # data contains:feature vector, state
    data = pd.read_csv(from_file, encoding='utf-8').dropna()
    test_nodes = data[['node1_x', 'node1_y', test_name, 'original']]
    test_matrix = test_nodes.to_numpy()
    origin_matrix = data[['node1_x', 'node1_y', 'original']].to_numpy()
    x_near = []
    y_near = []
    t_near = []
    for line in test_matrix:
        if line[2] == line[3]:
            x_near.append(line[0])
            y_near.append(line[1])
            t_near.append(line[3])
        else:
            xnear, ynear, tnear = find_near(line[0:3], origin_matrix)
            x_near.append(xnear)
            y_near.append(ynear)
            t_near.append(tnear)
    x = test_nodes[['node1_x']].to_numpy().T.reshape(num_nodes * K, )
    y = test_nodes[['node1_y']].to_numpy().T.reshape(num_nodes * K, )
    t = test_nodes[[test_name]].to_numpy().T.reshape(num_nodes * K, )
    x_near = np.asarray(x_near)
    y_near = np.asarray(y_near)
    t_near = np.asarray(t_near)
    accuracy_x = cal_accuracy(x, x_near)
    accuracy_y = cal_accuracy(y, y_near)
    accuracy_t = cal_accuracy(t, t_near)
    accuracy = (accuracy_x + accuracy_y + accuracy_t) / 3
    return accuracy


if __name__ == '__main__':

    from_file = process_data(save_path+'obs_net_hidden.csv',
                 save_path+'obs_1000x300_e_re_11071056.csv',
                 save_path+'true_net_net_hidden.csv')

    # from_file = '/Users/woffee/www/proposal/data/to_file_true_net_11051642_re.csv'
    print(from_file)

    K = 2
    num_nodes = 100
    test_name = 'time5'

    print(get_accuracy1(from_file, K, num_nodes, test_name))


