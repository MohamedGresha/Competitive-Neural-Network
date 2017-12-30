#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   vizualize.py
#=#| Date:   12/4/2017
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|

from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

import matplotlib
from matplotlib import pyplot
import numpy as np
import pickle
def plot_mnist(X, y, X_embedded, name, min_dist=10.0):
    theta = 7 * np.pi * np.random.rand(10000)

    colors = theta


    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(frameon=False)
    plt.title(r"textbf{MNIST dataset} -- Two-dimensional "
          "embedding of 70,000 handwritten digits with %s" % name)
    plt.setp(ax, xticks=(), yticks=())
    plt.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=0.9,
                    wspace=0.0, hspace=0.0)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
            c=colors, marker="x", cmap=plt.cm.RdGy)

    if min_dist is not None:
        from matplotlib import offsetbox
        shown_images = np.array([[15., 15.]])
        indices = np.arange(X_embedded.shape[0])
        np.random.shuffle(indices)
        for i in indices[:5000]:
            dist = np.sum((X_embedded[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist:
                continue
            shown_images = np.r_[shown_images, [X_embedded[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(X[i].reshape(28, 28),
                                      cmap='gray'), X_embedded[i])
            ax.add_artist(imagebox)

imageVFile = open('./Sources/images.pickle','rb')
imageVector = pickle.load(imageVFile)
imageLFile = open('./Sources/labels.pickle','rb')
labelVector = pickle.load(imageLFile)

imageVFile.close()
imageLFile.close()

imageVFile_Test = open('./Sources/images_Test.pickle','rb')
imageVector_Test = pickle.load(imageVFile_Test)
imageLFile_Test = open('./Sources/labels_Test.pickle','rb')
labelVector_Test = pickle.load(imageLFile_Test)

imageVFile_Test.close()
imageLFile_Test.close()


#
# x_pca = PCA(n_components=50).fit_transform(imageVector/ 255.0)
# x_train = x_pca[]
# x_train_embeded = TSNE(n_components=2, perplexity=40,verbose=2).fit_transform(x_train)
# plot_mnist(imageVector[:50000], labelVector)
#
#
# plt.show()



# from sklearn.datasets import fetch_mldata
#
# # Load MNIST dataset
# mnist = fetch_mldata("MNIST original")
# X, y = mnist.data / 255.0, mnist.target
#
# # Create subset and reduce to first 50 dimensions
# indices = np.arange(X.shape[0])
# np.random.shuffle(indices)
# n_train_samples = 5000
# X_pca = PCA(n_components=50).fit_transform(X)
# X_train = X_pca[indices[:n_train_samples]]
# y_train = y[indices[:n_train_samples]]


indices = np.arange(10000) # number of rows in X

data = np.matrix(np.divide(imageVector[:10000], 255.0)) # normalize by 255

y = np.matrix(labelVector[:60000])

x_pca = PCA(n_components=50).fit_transform(data[:10000])

x_train = x_pca[indices[:10000]]
y_train = y

# X_train_embedded = TSNE(n_components=2, perplexity=40, verbose=2).fit_transform(x_train)
#
# ff = open("viz.pickle",'wb')
# pickle.dump(X_train_embedded,ff)
#
# ff.close()

ff = open('viz.pickle','rb')

X_train_embedded = pickle.load(ff)
ff.close()


# indx = np.arange(10000)
# _lbls = np.array(labelVector)
#
# y_trainn = _lbls[indx[:5000]]

plot_mnist(data[indices[:60000]], y_train, X_train_embedded, "t-SNE", min_dist=20.0)
#plot_mnist(data[indices[:60000]], y_trainn, X_train_embedded, "t-SNE", min_dist=20.0)
plt.show()