"""


@Time    : 11/9/20
@Author  : Wenbo
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.special import binom
import matplotlib.cm as cmaps
import matplotlib.colors as colours
from mpl_toolkits.basemap import Basemap


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', help='file id', type=int, default=0)
args = parser.parse_args()



files = [
    'locations_e_ii_0.01.csv',
    'locations_e_ij_0.01.csv',
    'locations_e_ji_0.01.csv',
    'locations_e_jj_0.01.csv',
]
figure_titles = [
    'E_1,1',
    'E_1,2',
    'E_2,1',
    'E_2,2',
]
to_files = [
    'E_11_spatial.png',
    'E_12_spatial.png',
    'E_21_spatial.png',
    'E_22_spatial.png',
]

figure_title = figure_titles[ args.file ]
to_file = to_files[ args.file ]
data = pd.read_csv(files[args.file])

original = list(data.e)
hist, bins = np.histogram(original, bins=10, range=(0, 1))
print(hist)



fig, ax = plt.subplots(figsize=(12,6))


# m = Basemap(width=12000000,height=9000000,projection='lcc',
#             resolution=None,lat_1=45.,lat_2=55,lat_0=50,lon_0=-107.)
# m.shadedrelief()


cnorm1=colours.Normalize(vmin=0,vmax=1)
colormap1='Reds'
scalarmap1=cmaps.ScalarMappable(norm=cnorm1,cmap=colormap1)

points = []

print(len(data))

weights = np.zeros([100, 100])

cities = [
    [32.819859,-96.761754,'Dallas'],
    [37.774930,-122.419420,'San Francisco'],
    [33.767194,-84.433106,'Atlanta'],
    [42.358430,-71.059770,'Boston'],
    [41.850030,-87.650050,'Chicago'],
    [44.979970,-93.263840,'Minneapolis'],
    [39.952330,-75.163790,'Philadelphia'],
    [47.606210,-122.332070,'Seattle'],
    [25.774270,-80.193660,'Miami'],
    [39.739150,-104.984700,'Denver'],
    [29.763280,-95.363270,'Houston'],
    [36.174970,-115.137220,'Las Vegas'],
    [28.538340,-81.379240,'Orlando'],
    [42.331430,-83.045750,'Detroit'],
    [45.536402,-122.630909,'Portland'],
    [38.627270,-90.197890,'St. Louis'],
    [33.953350,-117.396160,'Los Angeles'],
    [21.319943,-157.799589,'Hawaii'],
]


def get_closest(p):
    threshold = 1.0
    for i,s in enumerate(cities):
        if abs(s[1]-p[0]) < threshold and abs(s[0]-p[1]) < threshold:
            return i
    print("error point:", p)
    exit()

# 归一化时，把每个图中的没条边权重都除四个图中所有边权重的最大值
# max_weight: 455.99999994817694
# max_weight: 503.99999992284853
# max_weight: 493.99999994721765
# max_weight: 668.9446494164807

max_weight = 668.9446494164807

for index, row in data.iterrows():
    if row['e'] < 0.1:
        continue

    i1 = get_closest( tuple( [row['long1'], row['lat1']] ) )
    i2 = get_closest( tuple( [row['long2'], row['lat2']] ) )
    if i1==i2:
        continue

    weights[i1][i2] = weights[i1][i2] + row['e']
    # max_weight = max(max_weight, weights[i1][i2])

print("max_weight:", max_weight)

arrow_list = []
total_points = len(cities)
for i in range(total_points):
    for j in range(total_points):
        if weights[i][j] > 0:
            arrow_list.append([weights[i][j], i, j])

arrow_list = sorted(arrow_list, key=lambda k: k[0] )

# 只画top100的箭头
for row in arrow_list[-100:]:
    weight, i, j = row
    w = weight / max_weight

    if w < 0.2:
        w = 0.2
    elif w < 0.4:
        w = 0.4
    elif w < 0.6:
        w = 0.6
    elif w < 0.8:
        w = 0.8
    else:
        w = 0.9

    rgba = scalarmap1.to_rgba( w )
    rgba = np.array(rgba)
    # rgba[rgba >= 1] = 0.99999
    # rgba[rgba <= 0] = 0.00001

    width = 2.

    alpha = 1
    # alpha = 1 * weights[i][j] / max_weight
    # min_alpha = 0.2
    # alpha = max(alpha, min_alpha)

    p1 = tuple([cities[i][1], cities[i][0]])
    p2 = tuple([cities[j][1], cities[j][0]])

    style = 'Simple,tail_width=1,head_width=4,head_length=8'
    kw = dict(arrowstyle=style, color=rgba, connectionstyle="angle3,angleA=90,angleB=0", alpha=alpha, shrinkA=5,
              shrinkB=8, patchA=None, patchB=None)
    arrow = mpatches.FancyArrowPatch(p1, p2, **kw)
    ax.add_patch(arrow)

# 只画top100的箭头，夏威夷由于权重过低，不需要画了
cities = cities[:-1]

# 画地点
x = np.array([ b for a,b,_ in cities])
y = np.array([ a for a,b,_ in cities])
plt.scatter(x, y, s=8, marker='o', c='gray', zorder=30)

# 画地名

city_names = [ r[2] for r in cities]
for i in range(len(cities)):
    plt.text( x[i], y[i]+0.5, city_names[i], horizontalalignment='center', zorder=40)

# points = sorted(points, key=lambda k: k[0])

# for i,lat,long,name in enumerate(cities):
#     print("%d  https://www.google.com/maps/@%.6f,%.6f,7z" % (i+1,p[1], p[0]) )

plt.xlim(min(x) - 5, max(x) + 5)
plt.ylim(min(y) - 5, max(y) + 5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title(figure_title, loc='left', fontweight="bold")
plt.savefig(to_file)
plt.show()
