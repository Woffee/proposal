#!/usr/bin/env python
# coding: utf-8
# Jan 24, 2020
# Wenbo


# from pandas import DataFrame, read_csv
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
# import string
import gensim
import pickle
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
# import logging
import queue
pd.set_option('display.max_columns', 500)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = BASE_DIR + '/data_twitter'

def read_data_from_xls(filepath):
    filelist = os.listdir(filepath)
    filelist = [item for item in filelist if item.endswith('.xls')]

    data = None
    for k, item in enumerate(filelist):
        print(item)
        data = pd.read_excel(filepath + item)
        break

    tweet = data[['TWEET_ID', 'CREATED_AT', 'FROM_USER', 'TEXT_',
                  'FOLLOWERS_COUNT', 'FRIENDS_COUNT', 'STATUSES_COUNT', 'lat', 'lon']]
    tweet['TWEET_ID'] = tweet['TWEET_ID'].map(lambda x: str(x))
    tweet['lon'] += 180
    tweet['CREATED_AT'] = tweet['CREATED_AT'].map(
        lambda x: int(str(x).split('T')[0].split(' ')[0].replace('-', '').replace('/', '')))
    for item in ['FOLLOWERS_COUNT', 'FRIENDS_COUNT', 'STATUSES_COUNT', 'lat', 'lon']:
        tweet[item] /= max(np.absolute(np.array(tweet[item])))
    # print(tweet.head())

    usernames = set(tweet.FROM_USER)
    print("len(usernames):", len(usernames))

    obs = []
    features = []
    date_list = [20170811 + i for i in range(21)] + [20170901 + i for i in range(11)]
    for username in usernames:
        userdata = tweet[tweet.FROM_USER == username]

        feature = userdata.iloc[0, 4:]
        features.append(list(feature))

        raw = []
        last_num = 0
        for d in date_list:
            today = 1 if len(userdata[userdata.CREATED_AT == d])>0 else 0
            num = today + last_num
            raw.append(num)
            last_num = num
        obs.append(raw)

        if len(features)>= 300:
            break


    obs = np.array(obs, dtype=np.int32)
    features = np.array(features, dtype=np.float)

    ss = np.sum(obs, axis=0)  # sum of cols
    zero_index = list(np.where(ss== 0)[0])
    obs = np.delete(obs, zero_index, axis=1) # delete cols which sum=0
    obs = obs.T

    print(features)
    # exit(0)

    np.savetxt(SAVE_PATH + '/obs.csv', obs, delimiter=',', fmt='%d')
    np.savetxt(SAVE_PATH + '/features.csv', features, delimiter=',', fmt='%.8f')
    # print(type(obs))
    # print(usernames)
    # print(type(tweet.loc[263, 'CREATED_AT']))

    # print(tweet.head())




def true_net_file():
    features_path = SAVE_PATH + '/features.csv'
    to_file = SAVE_PATH + '/true_net.csv'
    df = pd.read_csv(features_path, header=None)
    features = np.array(df)

    nodes = []
    for ff in features:
        for gg in features:
            nn = list(ff) + list(gg)
            nodes.append(nn)
            nodes.append(nn)
        for gg in features:
            nn = list(ff) + list(gg)
            nodes.append(nn)
            nodes.append(nn)
    true_net = pd.DataFrame(data=nodes, columns=['node1_1','node1_2','node1_3','node1_4','node1_5',
                                               'node2_1', 'node2_2', 'node2_3', 'node2_4', 'node2_5'])
    true_net.to_csv(to_file, index=None)


read_data_from_xls('/Users/woffee/www/xiaoqi_code/xiaoqi_data/twitter_data/')
true_net_file()
print("clean_data2 done")

