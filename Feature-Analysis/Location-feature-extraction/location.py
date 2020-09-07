# -*- coding:utf-8 -*-

from reader import Reader
from sklearn.decomposition import PCA
import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

import matplotlib.pyplot as plt
#np.set_printoptions(threshold='nan')


reader = Reader(range(2000, 2020, 20),2)#
reader.reader(Reader.create_directories(Reader.PREFIX,2.4, 121.8), [1])
local_domain, local_codomain ,files= reader.get_train_data(200)

X = np.reshape(local_domain,(len(local_domain),Reader.DATA_SIZE))
X = np.array(X)
print(X.shape)
pca = PCA(n_components=10)
pca.fit(X)
print(pca.explained_variance_ratio_)
max_vector = pca.components_[0]

eigValues = pca.explained_variance_.tolist()

#plt.plot(range(len(eigValues)), eigValues, "r*")
plt.plot(range(len(eigValues)), eigValues, "r*--", label="eigValues")
plt.xlabel('Nth principal component')
plt.ylabel('Eigenvalues')
plt.legend()
# plt.show()
plt.savefig("eigValues.eps")

###################################################
plt.figure()
x = np.arange(0, 100, 1)
y = np.arange(0, 40, 1)
X, Y = np.meshgrid(x, y)
Z = []
x_number = 100
for row in range(int(len(max_vector)/x_number)):
	Z.insert(0,[])
	for col in range(x_number):
		Z[0].append(max_vector[row* x_number+col])
###################################################

plt.contour(X, Y, Z, 40, linewidths=0.5)
plt.contourf(X, Y, Z, 40, cmap=plt.cm.jet)
plt.colorbar()
plt.savefig("eigVector.eps")
