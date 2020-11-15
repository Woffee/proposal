"""


@Time    : 11/14/20
@Author  : Wenbo
"""

# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# from matplotlib.ticker import MultipleLocator, FormatStrFormatter
# %matplotlib inline

# Success Rate when Choosing Different Time Steps
def Success_when_Choosing_Different_Time_Steps():

    x_obs = [80,90,100,110,120]
    success_rate_dense =   [0.815000,0.785000,0.725000,0.715000,0.560000]  # N=200
    success_rate_sparse = [0.815000 ,0.715000 ,0.710000 ,0.530000 ,0.475000 ] # N=200

    plt.figure(figsize=(10,5))
    # plt.title("Success Rate when Choosing Different Time Steps")
    plt.xlabel("Number of observations", fontsize=15)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.ylabel("Success rate", fontsize=15)
    plt.ylim(0, 1)
    plt.plot(x_obs, success_rate_dense,'-', color='blue', marker='o', label="Weighted dense network")
    plt.plot(x_obs, success_rate_sparse,'-',color='red',  marker='x',label="Weighted sparse network")
    plt.legend()
    plt.grid()
    plt.savefig('./Success_when_Choosing_Different_Time_Steps.eps', format='eps', dpi=1000)



# Accuracy when Choosing Different Time Steps
def Accuracy_when_Choosing_Different_Time_Steps():
    x_obs = [80,90,100,110,120]
    Accuracy_dense =   [0.997502, 0.997387, 0.997362, 0.996846, 0.991417] # N=200
    Accuracy_sparse = [0.997485655, 0.997290549, 0.997785218, 0.994571743, 0.99259612] # N=200

    plt.figure(figsize=(10,5))

    plt.xlabel("Number of observations", fontsize=15)
    plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
    plt.ylabel("Accuracy", fontsize=15)
    # plt.ylim(0, 1)
    plt.ylim(0.95, 1)
    plt.plot(x_obs, Accuracy_dense,'-', color='blue', marker='o', label="Weighted dense network")
    plt.plot(x_obs, Accuracy_sparse,'-',color='red',  marker='x',label="Weighted sparse network")
    plt.legend()
    plt.grid()
    plt.savefig('./Accuracy_when_Choosing_Different_Time_Steps.eps', format='eps', dpi=1000)



# Accuracy_when_Choosing_Different_N_and_M
def Accuracy_when_Choosing_Different_N_and_M():
    x_obs = [80,90,100,110,120]
    """

    """
    Accuracy_150 =   [0.995765,	0.993421,	0.992714,	0.990839,	0.988577]
    Accuracy_180 =   [0.997922,	0.995216,	0.994691,	0.991736,	0.989537]
    Accuracy_200 =   [0.996979,	0.996873,	0.994791,	0.9948,	0.992103]
    Accuracy_1000 =   [0.9978425,	0.997077472,	0.99619686,	0.995340696,	0.994231059]

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

    plt.plot(x_obs, Accuracy_1000,'-',color='purple',  marker='+',label="N = 1000")

    plt.legend()
    plt.grid()
    plt.savefig('./figures/Accuracy_when_Choosing_Different_N_and_M.eps', format='eps', dpi=1000)

# Success_rate_when_Choosing_Different_N_and_M
def Success_rate_when_Choosing_Different_N_and_M():
    x_obs = [80,90,100,110,120]
    Success_rate_150 =   [0.773333, 0.713333, 0.706667, 0.578667, 0.486666667]
    Success_rate_180 =   [0.823333, 0.757222, 0.651667, 0.654444, 0.588889]
    Success_rate_200 =   [0.8405, 0.7895, 0.726, 0.731, 0.549]
    Success_rate_1000 =   [0.702, 0.575, 0.5304, 0.4858, 0.4512]

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

    plt.plot(x_obs, Success_rate_1000,'-', color='purple',  marker='+',label="N = 1000")

    plt.legend()
    plt.grid()
    plt.savefig('./figures/Success_rate_when_Choosing_Different_N_and_M.eps', format='eps', dpi=1000)
    # plt.show()




if __name__ == '__main__':
    # Success_when_Choosing_Different_Time_Steps()
    # Accuracy_when_Choosing_Different_Time_Steps()
    Accuracy_when_Choosing_Different_N_and_M()
    Success_rate_when_Choosing_Different_N_and_M()
