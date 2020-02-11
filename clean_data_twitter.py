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

        tagged_data = [TaggedDocument(words=word_tokenize(str(_d).lower()), tags=[str(i)]) for i, _d in enumerate(data)]

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
            test_data = word_tokenize(str(d).lower())
            vecs.append( model.infer_vector(test_data) )
        return vecs

    def get_labels(self, data):
        vecs = self.get_vecs(data)
        y_pred = KMeans(n_clusters=K, random_state=9).fit_predict(vecs)
        return y_pred

# 感染状态只有 0 1
def to_obs_1(tweet, usernames, filename):
    obs = []
    features = []
    date_list = [20170811 + i for i in range(21)] + [20170901 + i for i in range(11)]
    T = len(date_list)
    for username in usernames:
        userdata = tweet[tweet.FROM_USER == username]

        status = []
        for d in date_list:
            today_data = userdata[userdata.CREATED_AT == d]
            if len(today_data)>0:
                status.append(1)
            else:
                status.append(0)

        # 感染状态持续 3 天
        # eg:  before: [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        #      after:  [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
        flag = 0
        res = [0] * T
        for i in range(T):
            if status[i] == 0 and flag > 0:
                res[i] = 1
                flag = flag - 1
            elif status[i] == 1:
                res[i] = 1
                flag = 2

        obs.append(res)

    obs = np.array(obs, dtype=np.int32)

    ss = np.sum(obs, axis=0)  # sum of cols
    zero_index = list(np.where(ss == 0)[0])
    obs = np.delete(obs, zero_index, axis=1)  # delete cols which sum=0
    obs = obs.T

    np.savetxt(filename, obs, delimiter=' ', fmt='%d')


# 感染数量 0 1 2 3
def to_obs_2(tweet, usernames, labels, filename, feature_filename):
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
                today_status[labels[index]] = 1
            status_0.append(today_status[0])
            status_1.append(today_status[1])

        for i in range(len(status_0)):
            if i > 0:
                status_0[i] += status_0[i - 1]
        for i in range(len(status_1)):
            if i > 0:
                status_1[i] += status_1[i - 1]

        obs.append(status_0)
        obs.append(status_1)

    obs = np.array(obs, dtype=np.int32)
    features = np.array(features, dtype=np.float)

    ss = np.sum(obs, axis=0)  # sum of cols
    zero_index = list(np.where(ss == 0)[0])
    obs = np.delete(obs, zero_index, axis=1)  # delete cols which sum=0
    obs = obs.T

    np.savetxt(filename, obs, delimiter=',', fmt='%d')
    np.savetxt(feature_filename, features, delimiter=',', fmt='%.8f')


def clean_date(date):
    try:
        return int(str(date).split('T')[0].split(' ')[0].replace('-', '').replace('/', ''))
    except:
        return 20170811

def read_data_from_xls(filepath):
    filelist = os.listdir(filepath)
    filelist = [item for item in filelist if item.endswith('.xls')]

    data = None
    for k, item in enumerate(filelist):
        print(item)
        now_data = pd.read_excel(filepath + item)
        if data is None:
            data = now_data
        else:
            data = pd.concat([data, now_data], sort=False)

    tweet = data[['TWEET_ID', 'CREATED_AT', 'FROM_USER', 'TEXT_',
                  'FOLLOWERS_COUNT', 'FRIENDS_COUNT', 'STATUSES_COUNT', 'lat', 'lon']]
    tweet['TWEET_ID'] = tweet['TWEET_ID'].map(lambda x: str(x))
    tweet['lon'] += 180

    tweet['CREATED_AT'] = tweet['CREATED_AT'].map(clean_date)

    for item in ['FOLLOWERS_COUNT', 'FRIENDS_COUNT', 'STATUSES_COUNT', 'lat', 'lon']:
        tweet[item] /= max(np.absolute(np.array(tweet[item])))
    # print(tweet.head())


    usernames = list(set(tweet.FROM_USER))
    usernames = [x for x in usernames if str(x) != 'nan']
    usernames = usernames[:300]

    print("usernames[:5] = ", usernames[:5])

    test_num = '5'

    # for NC
    to_obs_1(tweet, usernames, '/Users/woffee/www/ReconstructingNetwork/c_time_state_original_' + test_num + '.txt' )

    # tweets classifications
    d2v = MyDoc2vec()
    vecs = d2v.get_vecs(list(tweet.TEXT_))
    labels = KMeans(n_clusters=K, random_state=9).fit_predict(vecs)

    # for this project
    to_obs_2(tweet, usernames, labels, SAVE_PATH + '/obs_' + test_num + '.csv', SAVE_PATH + '/features_' + test_num + '.csv')

    # pca = PCA(n_components=2)
    # pca.fit(vecs)
    # X = pca.transform(vecs)
    #
    # fig, ax = plt.subplots()
    # ax.scatter(X[:, 0], X[:, 1], c=labels, alpha=0.5)
    # ax.grid(True)
    # fig.tight_layout()
    # plt.show()

# 从obs中，生成NC需要的数据
def recovery_process(original, days=3):
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
            if s > days:
                original[i][j] = 0
            # print(original[i][j])

        # 把数字改成1
        for i in range(original.shape[0]):
            if original[i][j]>1:
                original[i][j] = 1

    index = []
    for i in range(original.shape[0]):
        if sum(original[i])>0:
            index.append(i)
    return original[index]

def obs_to_nc(test_num, days = 3):
    obs = np.loadtxt(SAVE_PATH + ("/obs_%d.csv" % test_num) , delimiter=',')

    # aa = np.array([[1,2,3],[4,5,6],[7,8,9]])
    # print(sum(aa[1]))
    # exit()
    # print(aa.T)
    # print(aa[:, 1])
    # [2 5 8]

    obs_2 = []
    for i in range(obs.shape[1]):
        if i % 2==0:
            a = obs[:, i]
            b = obs[:, i+1]
            # print(a)
            # print(b)
            obs_2.append(a+b)
    obs_2 = np.array(obs_2).T
    obs_2 = recovery_process(obs_2, days)

    np.savetxt(SAVE_PATH + ("/c_time_state_original_%d_%d.txt" % (test_num,days)), obs_2, delimiter=' ', fmt='%d')


def true_net_file(test_num):
    features_path = SAVE_PATH + ("/features_%d.csv" % test_num)
    to_file = SAVE_PATH + ("/true_net_%d.csv" % test_num)
    print(to_file)
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

if __name__ == '__main__':
    for days in [3]:
        for i in range(6):
            if i>0:
                print(i, days)
                obs_to_nc(i, days)

    # read_data_from_xls('/Users/woffee/www/twitter_data/')
    # true_net_file()
    # print("clean_data_twitter done")

