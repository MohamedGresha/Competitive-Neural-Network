#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   DataVizulizer.py
#=#| Date:   12/5/2017
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|

import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA as sklearnPCA

imageVFile = open('./Sources/images.pickle', 'rb')
imageVector = pickle.load(imageVFile)
imageLFile = open('./Sources/labels.pickle', 'rb')
labelVector = pickle.load(imageLFile)
imageVFile.close()
imageLFile.close()


def createYArray(numSubClass, numClasses):
    assert numSubClass % numClasses == 0
    groups = int(numSubClass / numClasses)
    nums = groups * numClasses
    r = np.zeros(nums)
    cnt = 0
    for i in range(0, nums, groups):
        r[i:i + groups] = cnt

        cnt += 1

    return r
def displayWeights():
    def createYArray(numSubClass, numClasses):
        assert numSubClass % numClasses == 0
        groups = int(numSubClass / numClasses)
        nums = groups * numClasses
        r = np.zeros(nums)
        cnt = 0
        for i in range(0, nums, groups):
            r[i:i + groups] = cnt

            cnt += 1

        return r

    x = createYArray(50, 10)

    f = open("NetworkWeights.pick", 'rb')  # load the weights matrix which is a subclass X input vector
    prototypeV = pickle.load(f)
    X = prototypeV
    y = createYArray(150, 10)
    print(type(y))

    # normalize weights

    X_Norm = (X - X.min()) / (X.max() - X.min())

    pca = sklearnPCA(n_components=3)  # get 3 PCA's

    transformed = np.matrix(pca.fit_transform(X_Norm))  # transform the data

    fig = plt.figure(69)

    ax = Axes3D(fig)
    load_samples = True

    if load_samples:
        ff = open("PCA_10000.pick", 'rb')
        numberOfSamples = 10000  # needs to match the file name size
        sample_y = np.array(labelVector[:numberOfSamples])
        sample_Trans = pickle.load(ff)
        ff.close()

    # need to plot the scatual using 10K
    # for i in range(10):
    #     if load_samples:
    #         ax.scatter([sample_Trans[sample_y == i][:, 0]], [sample_Trans[sample_y == i][:, 1]], [sample_Trans[sample_y == i][:, 2]], alpha=.8,
    #                    label=i)
    #     ax.scatter(transformed[y==i][:,0], transformed[y==i][:,1], transformed[y==i][:,2], alpha=1,s = 100,marker='x',label=i)


    digit = 5
    digit2 = 9
    ax.scatter([sample_Trans[sample_y == digit][:, 0]], [sample_Trans[sample_y == digit][:, 1]],
               [sample_Trans[sample_y == digit][:, 2]], alpha=.2, label="Digit %s" % digit)
    ax.scatter(transformed[y == digit][:, 0], transformed[y == digit][:, 1], transformed[y == digit][:, 2], alpha=1,
               s=200, marker='x', label="Trained Weights %s" % digit)

    ax.scatter([sample_Trans[sample_y == digit2][:, 0]], [sample_Trans[sample_y == digit2][:, 1]],
               [sample_Trans[sample_y == digit2][:, 2]], alpha=.2, label="Digit %s" % digit2)
    ax.scatter(transformed[y == digit2][:, 0], transformed[y == digit2][:, 1], transformed[y == digit2][:, 2], alpha=1,
               s=100, marker='x', label="Trained Weights %s " % digit2)

    plt.title("Dataset Digits vs location of trained weights")
    plt.legend()
    plt.show()

def dataDriveWeights(subclasses, classes, inputSize):
    assert subclasses % classes == 0
    neuronsPerClass = int(subclasses / classes)
    import random
    #each neuron in the group should be a random input vector of the input

    data_X = np.matrix(imageVector)
    data_Y = np.array(labelVector)
    fives = data_X[data_Y ==5]
    print(len(fives))
    # for i in range(1,10):
    #     import random
    #     plt.subplot(3,3,i)
    #     idx = random.randint(0,len(fives))
    #     p = np.matrix(fives[idx]).reshape((28,28))
    #     plt.imshow(p)
    #
    # plt.show()
    d = np.zeros((subclasses, inputSize))


    #for the rows 0 -> 4 should be 0 digit 0
    digit = 0
    for i in range(0,subclasses, neuronsPerClass):
        ve = data_X[data_Y == digit]
        for j in range(i, i+neuronsPerClass):

            ridx = random.randint(0,len(ve))
            d[j,:] = ve[ridx]

        digit+=1


    for i in range(40,50):
        plt.subplot(2,5,i-39)
        p = d[i].reshape((28,28))
        plt.imshow(p)

    print(d.shape)
    plt.show()

    return d

import os
os.chdir(os.getcwd())
def displayDDI():
    ff = open("PCA_10000.pick", 'rb')
    numberOfSamples = 10000  # needs to match the file name size
    sample_y = np.array(labelVector[:numberOfSamples])
    sample_Trans = pickle.load(ff)
    ff.close()


    ff = open("withoutDDI//saverandominitWeight.p",'rb')
    initW = pickle.load(ff)
    ff.close()

    #trained weights without DDI
    ff = open("withoutDDI//NetworkWeights.pick",'rb')
    trainedWeights = pickle.load(ff)
    ff.close()

    #weights with DDI inital
    ff = open("NetworkWeights.pick","rb")
    trainedWeightsDDI = pickle.load(ff)
    ff.close()


    data = np.matrix(imageVector)
    data_y = np.array(labelVector)

    mean_0 = data[data_y == 0].mean(axis=0)
    mean_1 = data[data_y == 1].mean(axis=0)
    mean_2 = data[data_y == 2].mean(axis=0)
    mean_3 = data[data_y == 3].mean(axis=0)
    mean_4 = data[data_y == 4].mean(axis=0)
    mean_5 = data[data_y == 5].mean(axis=0)
    mean_6 = data[data_y == 6].mean(axis=0)
    mean_7 = data[data_y == 7].mean(axis=0)
    mean_8 = data[data_y == 8].mean(axis=0)
    mean_9 = data[data_y == 9].mean(axis=0)
    means = np.concatenate( ( mean_0,mean_1,mean_2,mean_3,mean_4,mean_5,mean_6,mean_7,mean_8,mean_9), axis=0)

    # get pca of this data which is 10X784

    X = means
    #Y = np.arange()
    #y = createYArray(150, 10)
    #print(type(y))

    # normalize weights

    X_Norm = (X - X.min()) / (X.max() - X.min())

    pca = sklearnPCA(n_components=3)  # get 3 PCA's

    transformed = np.matrix(pca.fit_transform(X_Norm))  # transform the data
    ########################
    initX = (initW - initW.min())/(initW.max() - initW.min())
    initWeightT =  np.matrix(pca.fit_transform(initX))
    initY = createYArray(100,10)
############################

    #######################
    trainedWX = trainedWeights
    tranedW_Norm = (trainedWX - trainedWX.min()) / (trainedWX.max() - trainedWX.min())
    trainedWW = np.matrix(pca.fit_transform(tranedW_Norm))

    #########################


    ################## trained iwht DDI Weights BEFORE

    trainedWDDI = trainedWeightsDDI
    trainedWDDINorm = (trainedWDDI - trainedWDDI.min()) / (trainedWDDI.max() - trainedWDDI.min())
    trainedWDDI = np.matrix(pca.fit_transform(trainedWDDINorm))
    ##############################################

    fig = plt.figure(69)

    ax = Axes3D(fig)
    colors = ['#D2691E','#FFF8DC','#00FFFF','#B8860B','#8B008B','#9932CC','#483D8B','#00BFFF','#DCDCDC','#008000']
    # for i in range(10):
    #     # ax.scatter(transformed[0][:,0], transformed[0][:,1], transformed[0][:,2], marker='x', s=100)
    #     # ax.scatter(sample_Trans[sample_y==0][:,0], sample_Trans[sample_y==0][:,1],sample_Trans[sample_y==0][:,2])
    #     ax.scatter(transformed[i][:,0], transformed[i][:,1], transformed[i][:,2], marker='x', s=300,c=colors[i],  label=i)
    #     ax.scatter(sample_Trans[sample_y==i][:,0], sample_Trans[sample_y==i][:,1],sample_Trans[sample_y==i][:,2], alpha=.2, s=40, c=colors[i],label=i)
    #

    digit1 = 6
    digit2 = 7
    digit3 = 5
    dataS = 10
    weightS = 500

    ax.scatter(initWeightT[initY==digit1][:, 0], initWeightT[initY==digit1][:, 1], initWeightT[initY==digit1][:, 2], marker='*', s=weightS,c='r', label="initalRandom %s " %digit1)
    ax.scatter(transformed[digit1][:,0], transformed[digit1][:,1], transformed[digit1][:,2], marker='x', s=weightS, c='r', label="DDI %s " % digit1)
    ax.scatter(sample_Trans[sample_y==digit1][:,0], sample_Trans[sample_y==digit1][:,1],sample_Trans[sample_y==digit1][:,2], c =colors[digit1], label=digit1, s=dataS)
    ax.scatter(trainedWW[initY == digit1][:, 0], trainedWW[initY == digit1][:, 1],trainedWW[initY == digit1][:, 2], marker='d', s=weightS, c='r', label="PostTrained PrototypeV %s " % digit1)
    ax.scatter(trainedWDDI[initY == digit1][:, 0], trainedWDDI[initY == digit1][:, 1],trainedWDDI[initY == digit1][:, 2], marker='v', s=weightS, c='r', label="trainedWDDI PrototypeV %s " % digit1)


    ax.scatter(initWeightT[initY==digit2][:, 0], initWeightT[initY==digit2][:, 1], initWeightT[initY==digit2][:, 2], marker='*', s=weightS,c='b', label="uniformRandom %s " % digit2)
    ax.scatter(transformed[digit2][:,0], transformed[digit2][:,1], transformed[digit2][:,2], marker='x', s=weightS, c='b', label="DDI %s " %digit2)
    ax.scatter(sample_Trans[sample_y==digit2][:,0], sample_Trans[sample_y==digit2][:,1],sample_Trans[sample_y==digit2][:,2], c =colors[digit2], label=digit2, s=dataS)
    ax.scatter(trainedWW[initY == digit2][:, 0], trainedWW[initY == digit2][:, 1],trainedWW[initY == digit2][:, 2], marker='d', s=weightS, c='b', label="PostTrained PrototypeV %s " % digit2)
    ax.scatter(trainedWDDI[initY == digit2][:, 0], trainedWDDI[initY == digit2][:, 1],trainedWDDI[initY == digit2][:, 2], marker='v', s=weightS, c='b', label="trainedWDDI PrototypeV %s " % digit2)

    #
    # ax.scatter(initWeightT[initY==digit3][:, 0], initWeightT[initY==digit3][:, 1], initWeightT[initY==digit3][:, 2], marker='*', s=weightS,c='g', label="uniformRandom %s " % digit3)
    # ax.scatter(transformed[digit3][:,0], transformed[digit3][:,1], transformed[digit3][:,2], marker='x', s=weightS, c='g', label="DDI %s " % digit3)
    # ax.scatter(sample_Trans[sample_y==digit3][:,0], sample_Trans[sample_y==digit3][:,1],sample_Trans[sample_y==digit3][:,2], c =colors[digit3+1], label=digit3, s=dataS)
    # ax.scatter(trainedWW[initY == digit3][:, 0], trainedWW[initY == digit3][:, 1],trainedWW[initY == digit3][:, 2], marker='d', s=weightS, c='g', label="PostTrained PrototypeV %s " % digit3)


    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_zlabel("PCA 3")

    plt.title(r"Using Data Driven Initialization $\eta$=%s" % numberOfSamples)
    plt.legend()

    plt.show()


def displayDataSetDistribution():

    #display the distributions of datset samples

    data_y = np.array(labelVector)
    plt.hist(data_y, label="%s samples" % len(data_y))
    plt.hist(data_y[:30000], label="%s Samples" % len(data_y[:30000]))
    plt.hist(data_y[:10000], label="%s Samples" % len(data_y[:10000]))
    plt.title("Sample Distribution of the given number of samples")
    plt.legend()
    plt.show()

def dispplayDistributionOfSingle():
    data = np.matrix(imageVector[0])
    plt.hist(data)
    plt.title("single sample distribution")
    plt.xlabel("Pixel density")
    plt.show()

#dispplayDistributionOfSingle()
#displayDataSetDistribution()
displayDDI()
#dataDriveWeights(50,10, 784)