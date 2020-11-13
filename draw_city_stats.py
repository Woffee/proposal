#!/usr/bin/env python
# coding: utf-8
"""
此外 在帮我做一个图 对于空间网络图中的每一个有连边的城市（四个自网络中只要有一个网络里这个城市连一条边就算） ，
把它内部用户的status count, friend count follower count计算均值方差，然后做成这样的图
https://matplotlib.org/3.1.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py

因为三个变量量纲不同，所以你可以取log再画，但是要把原始的数值想上面这个图一样标在柱子顶端

此外 每个主子加上errorbar 像这个一样 errorbar的长度表示方差

Nov 12, 2020
"""

import pandas as pd
import numpy as np
import os
import random

import matplotlib
import matplotlib.pyplot as plt
import pickle
import math


random.seed(10)
pd.set_option('display.max_columns', 500)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_PATH = BASE_DIR + '/data_twitter'
K = 2


city_names = {
    'Santa_Ana': '',
    'Dallas': 'Dallas',
    'Miami': 'Miami',
    'Phoenix': '',
    'Minneapolis': 'Minneapolis',
    'San_Francisco': 'San Francisco',
    'San_Diego': '',
    'Honolulu': '',
    'Denver': 'Denver',
    'Detroit': 'Detroit',
    'Philadelphia': 'Philadelphia',
    'St_Louis': 'St. Louis',
    'San_Antonio': '',
    'Las_Vegas': 'Las Vegas',
    'Atlanta': 'Atlanta',
    'Boston': 'Boston',
    'Portland': 'Portland',
    'Orlando': 'Orlando',
    'Seattle': 'Seattle',
    'Riverside': '',
    'Tampa': '',
    'Chicago': 'Chicago',
    'Houston': 'Houston',
}

cities_loc = [
    [32.819859,-96.761754,'Dallas'],
    [37.774930,-122.419420,'San Francisco'],
    [33.767194,-84.433106,'Atlanta'],
    [42.358430,-71.059770,'Boston'],
    [41.850030,-87.650050,'Chicago'],
    [44.979970,-93.263840,'Minneapolis'],
    [39.952330,-75.163790,'Philadelphia'],
    [47.606210,-122.332070,'Seattle'],
    [25.774270,-80.193660,'Miami'],
    # [39.739150,-104.984700,'Denver'],
    [29.763280,-95.363270,'Houston'],
    # [36.174970,-115.137220,'Las Vegas'],
    [28.538340,-81.379240,'Orlando'],
    [42.331430,-83.045750,'Detroit'],
    [45.536402,-122.630909,'Portland'],
    [38.627270,-90.197890,'St. Louis'],
    # [33.953350,-117.396160,'Los Angeles'],
    [21.319943,-157.799589,'Hawaii'],
]


def get_closest(lat, long):
    threshold = 2.0
    for i,s in enumerate(cities_loc):
        if abs(s[0]-lat) < threshold and abs(s[1]-long) < threshold:
            return s[2]
    print("error")
    return ''

def get_data_from_features():

    features = np.loadtxt("drawE/features_original_1002.csv", delimiter=',')
    detected_cities = []
    data1 = {}
    for i in range(features.shape[0]):
        lat = features[i][3]
        long = features[i][4]

        c = get_closest( lat, long )
        if c != "":
            if c not in data1.keys():
                data1[c] = {
                    "followers_count": [ features[i][0] ],
                    "friends_count": [ features[i][1] ],
                    "statuses_count": [features[i][2]],
                }
            else:
                data1[c]['followers_count'].append(features[i][0])
                data1[c]['friends_count'].append(features[i][1])
                data1[c]['statuses_count'].append(features[i][2])

    data = []
    for c in list(data1.keys()):
        data.append({
            'city': c,

            'statuses_mean': np.mean(data1[c]['statuses_count']),
            'statuses_var': np.var(data1[c]['statuses_count']),

            'friends_mean': np.mean(data1[c]['friends_count']),
            'friends_var': np.var(data1[c]['friends_count']),

            'followers_mean': np.mean(data1[c]['followers_count']),
            'followers_var': np.var(data1[c]['followers_count']),
        })
    return data





def clean_date(date):
    try:
        return int(str(date).split('T')[0].split(' ')[0].replace('-', '').replace('/', ''))
    except:
        return 20170811

def read_data_from_xls(filepath):
    filelist = os.listdir(filepath)
    filelist = [item for item in filelist if item.endswith('.xls')]

    data = []
    for k, item in enumerate(filelist):
        print(item)
        tweet = pd.read_excel(filepath + item)

        if 'CITY' not in tweet.columns:
            print("CITY not in ", item)
            continue
        city = list(tweet.CITY)[0]

        # if row['city'] in city_names and city_names[row['city']] != '':
        #     pass

        STATUSES_COUNT = list( tweet.STATUSES_COUNT )
        FRIENDS_COUNT = list( tweet.FRIENDS_COUNT )
        FOLLOWERS_COUNT = list( tweet.FOLLOWERS_COUNT )

        data.append({
            'city': city,

            'statuses_mean': np.mean(STATUSES_COUNT),
            'statuses_var': np.var(STATUSES_COUNT),

            'friends_mean': np.mean(FRIENDS_COUNT),
            'friends_var': np.var(FRIENDS_COUNT),

            'followers_mean': np.mean(FOLLOWERS_COUNT),
            'followers_var': np.var(FOLLOWERS_COUNT),
        })

    return data


# 绘图
data = get_data_from_features()
#data = read_data_from_xls('/Users/woffee/www/twitter_data/')


labels = []
statuses_mean = []
statuses_var = []

followers_mean = []
followers_var = []

friends_mean = []
friends_var = []

def my_log(a):
    return math.sqrt(a) / 10

for row in data:
    city_name = row['city']
    labels.append(city_name)

    statuses_mean.append(row['statuses_mean'])
    friends_mean.append(row['friends_mean'])
    followers_mean.append(row['followers_mean'])


    statuses_var.append(my_log(row['statuses_var']))
    followers_var.append(my_log(row['followers_var']))
    friends_var.append(my_log(row['friends_var']))
    
max_val = 0.0
max_val = max(max_val, np.max(statuses_mean))
max_val = max(max_val, np.max(followers_mean))
max_val = max(max_val, np.max(friends_mean))

x = np.arange(len(labels))  # the label locations
width = 0.25  # the width of the bars

fig, ax = plt.subplots(figsize=(12,8))
rects1 = ax.bar(x - width, statuses_mean, width, label='Statuses count', alpha=0.5, yerr=statuses_var, error_kw=dict(ecolor='gray', ealpha=0.5))
rects2 = ax.bar(x , followers_mean, width, label='Followers count' , alpha=0.5, yerr=followers_var, error_kw=dict(ecolor='gray', ealpha=0.5))
rects3 = ax.bar(x + width, friends_mean, width, label='Friends count', alpha=0.5, yerr=friends_var, error_kw=dict(ecolor='gray', ealpha=0.5))

# plt.errorbar(x, followers_mean, yerr=followers_var, uplims=True, label='uplims=True')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Mean')
ax.set_xlabel('City')
ax.set_title('City Statistics')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30)
ax.legend()

def autolabel(rects, data):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate("%.2f" % data[i],
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=6)

autolabel(rects1, statuses_mean)
autolabel(rects2, followers_mean)
autolabel(rects3, friends_mean)

fig.tight_layout()
plt.yscale('log')
plt.savefig("figures/city_stats.png")
# plt.show()

