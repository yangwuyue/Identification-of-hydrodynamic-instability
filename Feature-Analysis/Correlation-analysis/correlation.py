# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from reader_xin import Reader
import math
from scipy.spatial.distance import pdist
plt.figure(figsize=(8, 8))

reader = Reader(range(2000, 2010, 20), 1)#
reader.reader(Reader.create_directories(Reader.PREFIX,2.4, 121.8), [1])
local_domain, local_codomain, files = reader.get_train_data(200)
X = np.reshape(local_domain,(len(local_codomain), Reader.DATA_SIZE)).tolist()

print "size = ", len(X)

cov = []
for index in range(len(X)):
    cov.append([])
    # print "index = ", index
    for other in range(len(X)):
        # print "Other = ", other
        cov[-1].append(np.sum(np.abs(np.mat(X[index])-np.mat(X[other]))))

t_index = 0
max_dis = 0
for index in range(len(X)-1):
    if abs(cov[index][0]-cov[index+1][0]) > max_dis:
        t_index = index
        max_dis = abs(cov[index][0]-cov[index+1][0])

img = plt.matshow(cov,cmap=plt.cm.jet)
plt.colorbar(img)
label = [""]*len(files)
label[0] = files[0].split("-")[0]
label[t_index] = files[t_index].split("-")[0]
label[-1] = files[-1].split("-")[0]

	
plt.xticks(np.arange(len(X)), label, rotation=90)
plt.yticks(np.arange(len(X)), label)
#plt.title("r=0.6")
#plt.xlabel("Re")
#plt.ylabel("Re")
#plt.tight_layout()
#plt.savefig('RB.eps')
plt.show()
















