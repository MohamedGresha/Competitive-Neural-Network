#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   visualizeFun.py
#=#| Date:   12/5/2017
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

from sklearn.decomposition import PCA as sklearnPCA

from sklearn.discriminant_analysis import  LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs

from pandas.tools.plotting import parallel_coordinates
#
#
# # X, y = make_blobs(n_samples=200, centers=3, n_features=2, random_state=0)
# # print(y)
# # print(X)
# #
# # #plot
# #
# # plt.scatter(X[:,0],X[:,1], c=y)
# # plt.show()
#
# url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
# cols =  ['Class', 'Alcohol', 'MalicAcid', 'Ash', 'AlcalinityOfAsh', 'Magnesium', 'TotalPhenols',
#          'Flavanoids', 'NonflavanoidPhenols', 'Proanthocyanins', 'ColorIntensity',
#          'Hue', 'OD280/OD315', 'Proline']
# data = pd.read_csv(url, names=cols)
#
#
# #make the class labels in one variables
# #and the data in another
# y = data['Class']          # Split off classifications
# X = data.ix[:, 'Alcohol':] # Split off features
#
#
# X_norm = (X - X.min())/(X.max() - X.min())
#
#
# pca = sklearnPCA(n_components=2)
# transformed = pd.DataFrame(pca.fit_transform(X_norm))
#
# plt.scatter(transformed[y==1][0], transformed[y==1][1], label="C1",c='red')
# plt.scatter(transformed[y==2][0], transformed[y==2][1], label="C2",c='blue')
# plt.scatter(transformed[y==3][0], transformed[y==3][1], label="C3",c='lightgreen')
#
# plt.legend()
# plt.show()



import pickle
imageVFile = open('./Sources/images.pickle', 'rb')
imageVector = pickle.load(imageVFile)
imageLFile = open('./Sources/labels.pickle', 'rb')
labelVector = pickle.load(imageLFile)
imageVFile.close()
imageLFile.close()
imageVFile_Test = open('./Sources/images_Test.pickle', 'rb')
imageVector_Test = pickle.load(imageVFile_Test)
imageLFile_Test = open('./Sources/labels_Test.pickle', 'rb')
labelVector_Test = pickle.load(imageLFile_Test)
imageVFile_Test.close()
imageLFile_Test.close()


import numpy as np


numberOfSamples = 10000
X = np.matrix(imageVector[:numberOfSamples]) # get the x data sets
X_norm = (X - X.min())/ (X.max() - X.min()) # scale /normalize the weights

#pca = sklearnPCA(n_components=3)
#transformed = np.matrix(pca.fit_transform(X_norm))

y = np.array(labelVector[:numberOfSamples])

ff = open("PCA_10000.pick",'rb')

import pickle
transformed = pickle.load(ff)
ff.close()
print(transformed)

fig = plt.figure()
#ax = plt.axes(projection='3d')
ax = Axes3D(fig)
for i in range(10):
    ax.scatter([transformed[y == i][:, 0]], [transformed[y == i][:, 1]], [transformed[y == i][:, 2]], alpha=.8 , s =30, label=i)

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
ax.view_init(azim=-150)
# plt.scatter([transformed[y==1][:,0]], label="1")
# plt.scatter([transformed[y==2][:,0]],  label="2")
# plt.scatter([transformed[y==3][:,0]], label="3")
# plt.scatter([transformed[y==4][:,0]], label="4")
# plt.scatter([transformed[y==5][:,0]],  label="5")
# plt.scatter([transformed[y==6][:,0]],  label="6")
# plt.scatter([transformed[y==7][:,0]], label="7")
# plt.scatter([transformed[y==8][:,0]],  label="8")
# plt.scatter([transformed[y==9][:,0]], label="9")

# for i in range(10):
#     plt.scatter([transformed[y == i][:, 0]], [transformed[y == i][:, 1]], label=i)

# plt.scatter([transformed[y==1][:,0]], [transformed[y==1][:,1]], label="1")
# plt.scatter([transformed[y==2][:,0]], [transformed[y==2][:,1]], label="2")
# plt.scatter([transformed[y==3][:,0]], [transformed[y==3][:,1]], label="3")
# plt.scatter([transformed[y==4][:,0]], [transformed[y==4][:,1]], label="4")
# plt.scatter([transformed[y==5][:,0]], [transformed[y==5][:,1]],  label="5")
# plt.scatter([transformed[y==6][:,0]], [transformed[y==6][:,1]],  label="6")
# plt.scatter([transformed[y==7][:,0]], [transformed[y==7][:,1]], label="7")
# plt.scatter([transformed[y==8][:,0]], [transformed[y==8][:,1]],  label="8")
# plt.scatter([transformed[y==9][:,0]], [transformed[y==9][:,1]], label="9")
#


# plt.scatter([transformed[y==1][:,0]], [transformed[y==1][:,1]], [transformed[y==1][:,2]], label="1")
# plt.scatter([transformed[y==2][:,0]], [transformed[y==2][:,1]], [transformed[y==2][:,2]], label="2")
# plt.scatter([transformed[y==3][:,0]], [transformed[y==3][:,1]], [transformed[y==3][:,2]], label="3")
# plt.scatter([transformed[y==4][:,0]], [transformed[y==4][:,1]], [transformed[y==4][:,2]], label="4")
# plt.scatter([transformed[y==5][:,0]], [transformed[y==5][:,1]], [transformed[y==5][:,2]], label="5")
# plt.scatter([transformed[y==6][:,0]], [transformed[y==6][:,1]], [transformed[y==6][:,2]], label="6")
# plt.scatter([transformed[y==7][:,0]], [transformed[y==7][:,1]], [transformed[y==7][:,2]], label="7")
# plt.scatter([transformed[y==8][:,0]], [transformed[y==8][:,1]], [transformed[y==8][:,2]], label="8")
# plt.scatter([transformed[y==9][:,0]], [transformed[y==9][:,1]], [transformed[y==9][:,2]], label="9")

# ax.scatter([transformed[y==1][:,0]], [transformed[y==1][:,1]], [transformed[y==1][:,2]], label="1")
# ax.scatter([transformed[y==2][:,0]], [transformed[y==2][:,1]], [transformed[y==2][:,2]], label="2")
# ax.scatter([transformed[y==3][:,0]], [transformed[y==3][:,1]], [transformed[y==3][:,2]], label="3")
# ax.scatter([transformed[y==4][:,0]], [transformed[y==4][:,1]], [transformed[y==4][:,2]], label="4")
# ax.scatter([transformed[y==5][:,0]], [transformed[y==5][:,1]], [transformed[y==5][:,2]], label="5")
# ax.scatter([transformed[y==6][:,0]], [transformed[y==6][:,1]], [transformed[y==6][:,2]], label="6")
# ax.scatter([transformed[y==7][:,0]], [transformed[y==7][:,1]], [transformed[y==7][:,2]], label="7")
# ax.scatter([transformed[y==8][:,0]], [transformed[y==8][:,1]], [transformed[y==8][:,2]], label="8")


plt.title(r'3 PCA for MNIST dataset $\eta$=%s' % numberOfSamples)

plt.legend()
plt.show()