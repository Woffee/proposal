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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(BASE_DIR)

read_path = BASE_DIR + '/data/'
save_path = read_path

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
                                 'STATUSES_COUNT', 'TIME_ZONE', 'lon', 'lat']]  # 'PLACE_FULLNAME','PLACE_TYPE','CITY',
        user_info1.rename(columns={'FROM_USER': 'user_id'}, inplace=True)
        user_info = pd.concat([user_info, user_info1])
        tweet = pd.concat([tweet, data[['TWEET_ID', 'CREATED_AT', 'FROM_USER', 'LANGUAGE_', 'TEXT_']]])

tweet.rename(columns={'FROM_USER': 'user_id'}, inplace=True)
tweet['TWEET_ID'] = tweet['TWEET_ID'].map(lambda x: str(x))
tweet['CREATED_AT'] = tweet['CREATED_AT'].map(
    lambda x: int(str(x).split('T')[0].split(' ')[0].replace('-', '').replace('/', '')))
date_list = [20170811 + i for i in range(21)] + [20170901 + i for i in range(11)]
tweet['date_range'] = -1


# s.translate(None, string.punctuation)
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def remove_punctuation(input_txt):
    return re.sub(r'[^\w\s]', '', input_txt).lower()


tweet['tidy_tweet'] = np.vectorize(remove_pattern)(tweet['TEXT_'], "@[\w]*")
tweet['tidy_tweet'] = np.vectorize(remove_pattern)(tweet['tidy_tweet'], "https://t.co/[\w]*")
tweet['tidy_tweet'] = np.vectorize(remove_punctuation)(tweet['tidy_tweet'])


def read_text(text_list):
    i = -1
    for line in text_list:
        i = i + 1
        yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


train_data = list(read_text(tweet['tidy_tweet']))

print(train_data[:2])

filepath_doc2vec_model = save_path + 'doc2vec_model.bin'
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

# 2 Text to vectors
print("saving text to embeddings...")
test_docs = [x.strip().split() for x in tweet['tidy_tweet']]


# 3 k-means
print("k means...")
kmeans_path = save_path + "kmeans_model.pkl"

K = 3
if os.path.exists(kmeans_path):
    km = pickle.load(open(kmeans_path, "rb"))
else:
    X = []
    for d in test_docs:
        X.append(model.infer_vector(d))
    km = KMeans(n_clusters=K).fit(X)
    pickle.dump(km, open(kmeans_path, "wb"))



output_file = save_path + "text_embeddings.txt"
output = open(output_file, "w")
i = 0
for d in test_docs:
    label = str(km.labels_[i])
    output.write( label + "," + (" ".join([str(x) for x in model.infer_vector(d)]) + "\n"))
    i = i + 1
output.flush()
output.close()


tweet['text_label'] = km.labels_
print("km.labels:")
print(km.labels_[0:30])

# for k in range(K):
#     for d,date in enumerate(date_list):
#         df['k_' + str(k) + '_d_'+str(d)]=0
#         df[ df['text_label']==k ][ df['CREATED_AT']<=date]['k_' + str(k) + '_d_'+str(d)]=1

for k in range(K):
    for d, date in enumerate(date_list):
        tweet['k_' + str(k) + '_d_' + str(d)] = 0

df2 = None
for k in range(K):
    tmp = tweet[tweet['text_label'] == k]
    for d, date in enumerate(date_list):
        tmp['k_' + str(k) + '_d_' + str(d)] = 0
        tmp['k_' + str(k) + '_d_' + str(d)][tmp['CREATED_AT'] <= date] = 1
    if k == 0:
        df2 = tmp
    else:
        df2 = pd.concat([df2, tmp])

key = ['user_id']
for k in range(K):
    for d, date in enumerate(date_list):
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
input_data.to_csv( save_path + 'input.csv', index=False, encoding='utf-8')

print("done")
