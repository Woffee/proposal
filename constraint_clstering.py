#!/usr/bin/env python
# coding: utf-8

from numpy import *
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import datetime

import matplotlib.pyplot as plt
import numpy as np
import gensim
import pickle
import os
import copy

from scipy.spatial import distance_matrix


# In[2]:
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

save_path = BASE_DIR + '/data/'
embedding_path = save_path + "text_embeddings.txt"

labels = []
vectors = []
with open(embedding_path, "r") as file:
    i = 0
    for line in file:
#         if i>50:
#             break
#         i = i+1

        arr = line.split(',', 1)
        label = int(arr[0])
        labels.append(label)
        vec = map(lambda x:float(x), arr[1].strip().split(' '))
        vectors.append(vec)

df = pd.DataFrame({
    'label':labels,
    'vector': vectors
})
# print df

kmeans_path = save_path + "kmeans_model.pkl"
if not os.path.exists(kmeans_path):
    print("get kmeans model error")
    exit()

km = pickle.load(open(kmeans_path, "rb"))
centers = km.cluster_centers_


# In[3]:


# print int(df[df['label']==1].head().index[0])

test = df.head(10000)


# In[4]:


def getC(df, centers):
    C = []
    K = 3
    for l in range(K):
        C.append(fi(df, centers, l))
    return C


# In[5]:


def fi(df, centers, l):
    c = centers[l]
    nodes = df[ df['label']==l ]['vector']
    s = 0
    for n in nodes:
        x = np.array(n, dtype=float)
        s = s + np.sum((x-c)**2)
    return s


# In[6]:


def fi2(x, centers, origin, des):
    c1 = centers[origin]
    c2 = centers[des]
    s1 = np.sum((x-c1)**2)
    s2 = np.sum((x-c2)**2)
    return s2<s1


# In[7]:


def Switching3(df, labels, clusters, centers, C):
    # 计算每个点到中心的距离矩阵
    print("Calculating dm...")
    dm = np.zeros( (len(df), len(centers)), dtype = float)

    for i in range(len(df)):
        for j in range(len(centers)):
            tmp = ( df.iloc[i]['vector'] - centers[j] )**2
            dm[i,j] = tmp.sum()

    
    print("Getting order...")
    order = []
    K = 3
    for ji in range( K ):
        set_i = df[ df['label']==ji ].index
        for j in range(K):
            if j == ji:
                continue
            for i in set_i:
                if dm[i,j] < dm[i,ji]:
                    order.append( [i,ji,j,dm[i,j]] )

    print('len of order: ',len(order))
    print('Sorting order...')
    order.sort(key=lambda x:x[3]) 

    i = 0
    for item in order:
        origin = item[1]
        des    = item[2]
        id_    = item[0]
        
        print(id_, origin, des)
        
        if fi2(df.iloc[id_]['vector'], centers, origin, des):
            print('updated')
            df.set_value(id_,'label', des)
            
#         print 'cc=',cc
#         if cc < C[des]:    
#             # reset cli = fi()
#             C[des] = cc
#             print 'updated'
#         else:
#             df.set_value(id_,'label', origin)
#             print 'reverted'
    return df
                


# In[8]:


print('getting C...')
C = getC(df, centers)
print(C)


# In[9]:


print('Switching3...')
tt = Switching3(test, labels, [], centers, C)


# In[ ]:




