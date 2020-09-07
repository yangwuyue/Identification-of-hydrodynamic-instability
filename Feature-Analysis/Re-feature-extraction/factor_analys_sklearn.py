# -*- coding:utf-8 -*-

import math
import numpy as np
import numpy.linalg as nlg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from reader_xin import Reader

reader = Reader(range(2000, 2010, 10),2)#
directories = Reader.create_directories(Reader.PREFIX, 2.4, 121.8)
reader.reader(directories, [1])
local_domain, local_codomain,t_files = reader.get_train_data(200)
local_domain = np.reshape(local_domain, (len(local_codomain), 200*Reader.ROW_NUMBER*1)).tolist()
for index in range(len(local_domain)):
	local_domain[index]=local_domain[index]
data = np.array(local_domain)

print(len(data))
print(len(data[0]))

from sklearn.decomposition import FactorAnalysis

fa = FactorAnalysis(n_components=2)
reduce_data = fa.fit_transform(data).tolist()
features = reduce_data

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(features)
labels = kmeans.labels_.tolist()

print(labels)

features_map = {}
re_map = {}
for index in range(len(labels)):
    if labels[index] not in features_map:
        features_map[labels[index]] = []
        re_map[labels[index]] = []
    features_map[labels[index]].append(features[index])
    re_map[labels[index]].append(directories[index])

from matplotlib import pyplot as plt
f = plt.figure(figsize=(5, 5))
ax = f.add_subplot(111)
colors = ['r', 'b', 'g', 'k', 'y']
t = [0,1,2]
for key in t:
    features = np.matrix(features_map[key])
    ax.scatter(features[:,0].tolist(), features[:, 1].tolist(), s=20,c=colors[key])

ax.set_title("Factor Analysis 3 Components")
ax.set_xlabel('First factor')
ax.set_ylabel('Second factor')
plt.show()

for key in re_map:
    print("class id %d" % key)
    print(re_map[key])

for key in features_map:
    print (len(features_map[key]))
    print (len(re_map[key]))
