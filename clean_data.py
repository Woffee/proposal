#!/usr/bin/env python
# coding: utf-8


from pandas import DataFrame, read_csv
# import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
import string
import gensim
import pickle
import os
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.cluster import KMeans
import logging

class Clean_data():
    def __init__(self, save_path, K):
        self.date_list = [20170811 + i for i in range(21)] + [20170901 + i for i in range(11)]
        self.K = K
        self.save_path = save_path


    # s.translate(None, string.punctuation)
    def remove_pattern(self, input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i, '', input_txt)
        return input_txt

    def remove_punctuation(self, input_txt):
        return re.sub(r'[^\w\s]', '', input_txt).lower().strip()

    def read_text(self, text_list):
        i = -1
        for line in text_list:
            i = i + 1
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

    def read_data_from_xls(self, read_path):
        filelist = os.listdir(read_path)
        filelist = [item for item in filelist if item.endswith('.xls')]

        print("reading xls...")
        tweet_num = 0
        for k, item in enumerate(filelist):
            if k == 0:
                data = pd.read_excel(read_path + item)
                print(k, read_path + item)
                # user_id=array(data['FROM_USER'])
                user_info = data.groupby('FROM_USER', as_index=False).first()
                user_info = user_info[['FROM_USER', 'FROM_USER_NAME', 'LOCATION', 'FOLLOWERS_COUNT', 'FRIENDS_COUNT',
                                       'STATUSES_COUNT', 'TIME_ZONE',
                                       'lon', 'lat']]  # 'PLACE_FULLNAME','PLACE_TYPE','CITY',
                user_info.rename(columns={'FROM_USER': 'user_id'}, inplace=True)
                tweet = data[['TWEET_ID', 'CREATED_AT', 'FROM_USER', 'LANGUAGE_', 'TEXT_']]
            else:
                data = pd.read_excel(read_path + item)
                print(k, read_path + item)
                # user_id=append(user_id,array(data['FROM_USER']))
                user_info1 = data.groupby('FROM_USER', as_index=False).first()
                user_info1 = user_info1[['FROM_USER', 'FROM_USER_NAME', 'LOCATION', 'FOLLOWERS_COUNT', 'FRIENDS_COUNT',
                                         'STATUSES_COUNT', 'TIME_ZONE', 'lon',
                                         'lat']]  # 'PLACE_FULLNAME','PLACE_TYPE','CITY',
                user_info1.rename(columns={'FROM_USER': 'user_id'}, inplace=True)
                user_info = pd.concat([user_info, user_info1])
                tweet = pd.concat([tweet, data[['TWEET_ID', 'CREATED_AT', 'FROM_USER', 'LANGUAGE_', 'TEXT_']]])

        tweet.rename(columns={'FROM_USER': 'user_id'}, inplace=True)
        tweet['TWEET_ID'] = tweet['TWEET_ID'].map(lambda x: str(x))
        tweet['CREATED_AT'] = tweet['CREATED_AT'].map(
            lambda x: int(str(x).split('T')[0].split(' ')[0].replace('-', '').replace('/', '')))

        tweet['date_range'] = -1

        tweet['tidy_tweet'] = np.vectorize(self.remove_pattern)(tweet['TEXT_'], "@[\w]*")
        tweet['tidy_tweet'] = np.vectorize(self.remove_pattern)(tweet['tidy_tweet'], "https://t.co/[\w]*")
        tweet['tidy_tweet'] = np.vectorize(self.remove_punctuation)(tweet['tidy_tweet'])

        print("remove empty")
        tweet = tweet[tweet['tidy_tweet'] != '']

        return tweet, user_info


    def tweet2vec(self, tweet):
        train_data = list(self.read_text(tweet['tidy_tweet']))

        print(train_data[:2])

        filepath_doc2vec_model = self.save_path + 'doc2vec_model.bin'
        if os.path.exists(filepath_doc2vec_model):
            print("loading model: ", filepath_doc2vec_model)
            model = Doc2Vec.load(filepath_doc2vec_model)
        else:
            print("training model...")
            model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
            model.build_vocab(train_data)
            # 1 Train model
            model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)
            # get_ipython().magic(u'time model.train(train_data, total_examples=model.corpus_count, epochs=model.epochs)')
            print("saving model...")
            model.save(filepath_doc2vec_model)

        test_docs = [x.strip().split() for x in tweet['tidy_tweet']]

        # k-means
        print("k means...")
        kmeans_path = self.save_path + "kmeans_model.pkl"


        if os.path.exists(kmeans_path):
            km = pickle.load(open(kmeans_path, "rb"))
        else:
            X = []
            for d in test_docs:
                X.append(model.infer_vector(d))
            km = KMeans(n_clusters= self.K).fit(X)
            pickle.dump(km, open(kmeans_path, "wb"))

        print("saving text to embeddings...")
        output_file = self.save_path + "text_embeddings.txt"
        output = open(output_file, "w")
        i = 0
        for d in test_docs:
            label = str(km.labels_[i])
            output.write(label + "," + (" ".join([str(x) for x in model.infer_vector(d)]) + "\n"))
            i = i + 1
        output.flush()
        output.close()

        tweet['text_label'] = km.labels_
        self.centers = km.cluster_centers_

        print("km.labels:")
        print(km.labels_[0:30])
        return tweet


    def output(self, tweet, user_info):
        # for k in range(K):
        #     for d,date in enumerate(date_list):
        #         df['k_' + str(k) + '_d_'+str(d)]=0
        #         df[ df['text_label']==k ][ df['CREATED_AT']<=date]['k_' + str(k) + '_d_'+str(d)]=1

        for k in range(self.K):
            for d, date in enumerate(self.date_list):
                tweet['k_' + str(k) + '_d_' + str(d)] = 0

        df2 = None
        for k in range(self.K):
            tmp = tweet[tweet['text_label'] == k]
            for d, date in enumerate(self.date_list):
                tmp['k_' + str(k) + '_d_' + str(d)] = 0
                tmp['k_' + str(k) + '_d_' + str(d)][tmp['CREATED_AT'] <= date] = 1
            if k == 0:
                df2 = tmp
            else:
                df2 = pd.concat([df2, tmp])

        key = ['user_id']
        for k in range(self.K):
            for d, date in enumerate(self.date_list):
                key.append('k_' + str(k) + '_d_' + str(d))

        # df2[df2['FROM_USER']=='LuncefordLee'][key]
        # df2.head(20)[key]

        print(len(df2))
        df3 = df2[key].groupby('user_id', as_index=False).agg(sum)

        user_info = user_info.groupby('user_id', as_index=False).first()
        user_info = pd.merge(user_info, df3, on='user_id', how='inner')

        # print len(df3)
        # user_info.head(20)

        if 'user_id' in key:
            key.remove('user_id')

        print("saving to input.csv...")
        input_data = user_info[['user_id', 'FOLLOWERS_COUNT', 'FRIENDS_COUNT', 'STATUSES_COUNT', 'lat', 'lon'] + key]
        input_data.to_csv(self.save_path + 'input.csv', index=False, encoding='utf-8')

    # =====================================
    def fi(self, df, centers, l):
        c = centers[l]
        nodes = df[df['label'] == l]['vector']
        s = 0
        for n in nodes:
            x = np.array(n, dtype=float)
            s = s + np.sum((x - c) ** 2)
        return s

    def fi2(self, x, centers, origin, des):
        c1 = centers[origin]
        c2 = centers[des]
        s1 = np.sum((x - c1) ** 2)
        s2 = np.sum((x - c2) ** 2)
        return s2 < s1

    def getC(self, df, centers):
        C = []

        for l in range(self.K):
            C.append(self.fi(df, centers, l))
        return C

    def switching3(self, df, centers, C):
        # 计算每个点到中心的距离矩阵
        print("Calculating dm...")
        dm = np.zeros((len(df), len(centers)), dtype=float)

        for i in range(len(df)):
            for j in range(len(centers)):
                tmp = (df.iloc[i]['vector'] - centers[j]) ** 2
                dm[i, j] = tmp.sum()

        print("Getting order...")
        order = []

        for ji in range(self.K):
            set_i = df[df['label'] == ji].index
            for j in range(self.K):
                if j == ji:
                    continue
                for i in set_i:
                    # print(dm[i,j])
                    # print(dm[i, ji])
                    if dm[i, j] < dm[i, ji]:
                        order.append([i, ji, j, dm[i, j]])

        print('len of order: ', len(order))
        print('Sorting order...')
        order.sort(key=lambda x: x[3])

        i = 0
        for item in order:
            origin = item[1]
            des = item[2]
            id_ = item[0]

            print(id_, origin, des)

            if self.fi2(df.iloc[id_]['vector'], centers, origin, des):
                print('updated')
                df.set_value(id_, 'label', des)

        #         print 'cc=',cc
        #         if cc < C[des]:
        #             # reset cli = fi()
        #             C[des] = cc
        #             print 'updated'
        #         else:
        #             df.set_value(id_,'label', origin)
        #             print 'reverted'
        return df

    def update_classification(self, embedding_path, tweet):
        labels = []
        vectors = []
        with open(embedding_path, "r") as file:
            i = 0
            for line in file:
                arr = line.split(',', 1)
                label = int(arr[0])
                labels.append(label)
                vec = list(map(float, arr[1].strip().split(' ')))
                vectors.append(vec)

        df = pd.DataFrame({
            'label': labels,
            'vector': vectors
        })

        df = self.switching3(df, self.centers, [])

        tweet['text_label'] = df['label']
        return tweet


    def init_classification(self):
        tweet, user_info = self.read_data_from_xls(self.save_path)
        tweet = self.tweet2vec(tweet)
        self.output(tweet, user_info)
        # print("done")
        return tweet, user_info

if __name__ == '__main__':
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print(BASE_DIR)

    save_path = BASE_DIR + '/data/'

    clean_data = Clean_data(save_path, K=3)
    clean_data.init_classification()
    clean_data.update_classification(save_path + "text_embeddings.txt", {})
