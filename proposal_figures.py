# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# %matplotlib inline

# Success Rate when Choosing Different Time Steps
def Success_when_Choosing_Different_Time_Steps():

    x_obs = [80,90,100,110,120]
    success_rate_dense =   [0.997502,0.997387,0.997362,0.996846,0.991417] # N=200
    success_rate_sparse = [0.997485655,0.997290549,0.997785218,0.994571743,0.99259612] # N=200

    plt.figure(figsize=(10,5))
    # plt.title("Success Rate when Choosing Different Time Steps")
    plt.xlabel("Number of observations", fontsize=15)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.ylabel("Absolute success rate", fontsize=15)
    plt.ylim(0.95, 1)
    plt.plot(x_obs, success_rate_dense,'-', color='blue', marker='o', label="Weighted dense network")
    plt.plot(x_obs, success_rate_sparse,'-',color='red',  marker='x',label="Weighted sparse network")
    plt.legend()
    plt.grid()
    plt.savefig('./Success_when_Choosing_Different_Time_Steps.eps', format='eps', dpi=1000)



# Accuracy when Choosing Different Time Steps
def Accuracy_when_Choosing_Different_Time_Steps():
    x_obs = [80,90,100,110,120]
    Accuracy_dense =   [0.815000,0.785000,0.725000,0.715000,0.560000] # N=200
    Accuracy_sparse = [0.815000 ,0.715000 ,0.710000 ,0.530000 ,0.475000 ] # N=200

    plt.figure(figsize=(10,5))

    plt.xlabel("Number of observations", fontsize=15)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.ylabel("Accuracy", fontsize=15)
    plt.ylim(0, 1)
    plt.plot(x_obs, Accuracy_dense,'-', color='blue', marker='o', label="Weighted dense network")
    plt.plot(x_obs, Accuracy_sparse,'-',color='red',  marker='x',label="Weighted sparse network")
    plt.legend()
    plt.grid()
    plt.savefig('./Accuracy_when_Choosing_Different_Time_Steps.eps', format='eps', dpi=1000)



# Accuracy_when_Choosing_Different_N_and_M
def Accuracy_when_Choosing_Different_N_and_M():
    x_obs = [80,90,100,110,120]

    Accuracy_150 =   [0.994549,0.993953,0.992969,0.992132,0.985658]

    Accuracy_170 =   [0.994128,0.995132,0.994417,0.989809,0.993624]
    Accuracy_180 =   [0.997582,0.995895,0.995703,0.992306,0.993610]
    Accuracy_190 =   [0.999459,0.995479,0.996700,0.994257,0.992622]
    Accuracy_200 =   [0.997502,0.997387,0.997362,0.996846,0.991417]


    x_obs_1000 = [80, 100]
    Accuracy_1000 =   [0.998946404, 0.997614]

    plt.figure(figsize=(10,5))

    plt.xlabel("Number of observations", fontsize=15)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.ylabel("Accuracy", fontsize=15)
    plt.ylim(0.95, 1)

    plt.plot(x_obs, Accuracy_150,'-', color='blue', marker='o', label="N = 150")
    # plt.plot(x_obs, Accuracy_170,'-', color='blue', marker='o', label="N = 170")
    plt.plot(x_obs, Accuracy_180,'-',color='red',  marker='x',label="N = 180")
    # plt.plot(x_obs, Accuracy_190,'-',color='yellow',  marker='v',label="N = 190")
    plt.plot(x_obs, Accuracy_200,'-',color='green',  marker='s',label="N = 200")

    plt.plot(x_obs_1000, Accuracy_1000,'-',color='purple',  marker='+',label="N = 1000")

    plt.legend()
    plt.grid()
    plt.savefig('./Accuracy_when_Choosing_Different_N_and_M.eps', format='eps', dpi=1000)

# Success_rate_when_Choosing_Different_N_and_M
def Success_rate_when_Choosing_Different_N_and_M():
    x_obs = [80,90,100,110,120]
    Success_rate_150 =   [0.733333,0.680000,0.620000,0.513333,0.460000]
    Success_rate_170 =   [0.723529,0.729412,0.652941,0.552941,0.570588]
    Success_rate_180 =   [0.772222,0.716667,0.655556,0.655556,0.672222]
    Success_rate_190 =   [0.873684,0.694737,0.673684,0.600000,0.526316]
    Success_rate_200 =   [0.815000,0.785000,0.725000,0.715000,0.560000]

    x_obs_1000 = [80,100]
    Success_rate_1000 =   [0.702,0.503000]

    plt.figure(figsize=(10,5))

    plt.xlabel("Number of observations", fontsize=15)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.ylabel("Success rate", fontsize=15)
    plt.ylim(0, 1)
    plt.plot(x_obs, Success_rate_150,'-', color='blue', marker='o', label="N = 150")
    # plt.plot(x_obs, Success_rate_170,'-', color='blue', marker='o', label="N = 170")
    plt.plot(x_obs, Success_rate_180,'-', color='red',  marker='x',label="N = 180")
    # plt.plot(x_obs, Success_rate_190,'-', color='yellow',  marker='v',label="N = 190")
    plt.plot(x_obs, Success_rate_200,'-', color='green',  marker='s',label="N = 200")

    plt.plot(x_obs_1000, Success_rate_1000,'-', color='purple',  marker='+',label="N = 1000")

    plt.legend()
    plt.grid()
    plt.savefig('./Success_rate_when_Choosing_Different_N_and_M.eps', format='eps', dpi=1000)




if __name__ == '__main__':
    # Success_when_Choosing_Different_Time_Steps()
    # Accuracy_when_Choosing_Different_Time_Steps()
    Accuracy_when_Choosing_Different_N_and_M()
    # Success_rate_when_Choosing_Different_N_and_M()
