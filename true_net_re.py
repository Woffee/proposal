import pandas as pd
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_path = BASE_DIR + '/data/'

rundate = time.strftime("%m%d%H%M", time.localtime())

true_net_filepath = save_path + "true_net_100x100_original.csv"
E_filepath = save_path + "to_file_E_11121605.csv"
true_net_re_filepath = save_path + "to_file_true_net_" + rundate + "_re.csv"

true_net = pd.read_csv(true_net_filepath, sep=',')
# true_net.drop([true_net.columns[0]], axis=1, inplace=True)

E = pd.read_csv(E_filepath, sep=',', header=None)

print("true_net.shape", true_net.shape)
print("E.shape", E.shape)

true_net['e'] = E.values.flatten()
true_net.to_csv(true_net_re_filepath, header=True, index=None)
print(true_net_re_filepath)