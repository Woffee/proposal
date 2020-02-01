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
from nltk.tokenize import word_tokenize
import nltk
# nltk.download('punkt')


pd.set_option('display.max_columns', 500)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = BASE_DIR + '/data_twitter'
K = 2

class MyDoc2vec():

    def train(self, data):
        # data = ["I love machine learning. Its awesome.",
        #         "I love coding in python",
        #         "I love building chatbots",
        #         "they chat amagingly well"]

        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]

        max_epochs = 10
        vec_size = 300
        alpha = 0.025

        model = Doc2Vec(size=vec_size,
                        alpha=alpha,
                        min_alpha=0.00025,
                        min_count=2,
                        dm=1)
        model.build_vocab(tagged_data)
        print("training d2v model...")
        for epoch in range(max_epochs):
            print('iteration {0}'.format(epoch))
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

        model.save(SAVE_PATH + "/d2v.model")
        return model

    def get_vecs(self, data):
        if os.path.exists(SAVE_PATH + "/d2v.model"):
            model = Doc2Vec.load(SAVE_PATH + "/d2v.model")
        else:
            model = self.train(data)

        vecs = []
        for d in data:
            test_data = word_tokenize(d.lower())
            vecs.append( model.infer_vector(test_data))
        return vecs

    def get_labels(self, data):
        vecs = self.get_vecs(data)
        y_pred = KMeans(n_clusters=K, random_state=9).fit_predict(vecs)
        return y_pred


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

    d2v = MyDoc2vec()
    labels = d2v.get_labels(list(tweet.TEXT_))

    obs = []
    features = []
    date_list = [20170811 + i for i in range(21)] + [20170901 + i for i in range(11)]
    for username in usernames:
        userdata = tweet[tweet.FROM_USER == username]

        feature = userdata.iloc[0, 4:]
        features.append(list(feature))

        status_0 = []
        status_1 = []

        for d in date_list:
            today_data = userdata[userdata.CREATED_AT == d]
            today_status = [0, 0]
            for index, row in today_data.iterrows():
                today_status[ labels[index] ] = 1
            status_0.append(today_status[0])
            status_1.append(today_status[1])

        for i in range(len(status_0)):
            if i>0:
                status_0[i] += status_0[i-1]
        for i in range(len(status_1)):
            if i>0:
                status_1[i] += status_1[i-1]

        obs.append(status_0)
        obs.append(status_1)

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


read_data_from_xls('/Users/woffee/www/twitter_data/')
true_net_file()
print("clean_data_twitter done")

