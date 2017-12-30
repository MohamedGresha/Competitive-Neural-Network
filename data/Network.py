

from mnist import MNIST
import numpy as np
import pickle, os, random, time
from matplotlib import pyplot as plt
from neurolab.trans import Competitive
plt.ion()
plt.axhline(linewidth=2, color='black')
plt.axvline(linewidth=2, color='black')
plt.grid()
plt.xlim([-1,1])
plt.ylim([-1,1])
class Network:

    # //classification on zip(//define how many bits for network IsADirectoryError//
    # //#bits for network id
    # patrica tree given the buts ones and zeros of address and then made aroutingtable
    # then made to find a particular host in a network
    # then refily talked abut IGP stop RIP and OSPF how routing tables are propagated

    #  BGP being able to reach something rather than reach it fast

    #routing, Automos sstems(AS) arbitrary defined,
        # ONe page one sided hand written notes for final

    def __init__(self, epoach, learning_rate= .7, verbose = False):
        """
        Default constructor
        """

        self.layers = []                    # each index is a layer
        self.learning_rate = learning_rate  # the learning tate of the network
        self.epoach = epoach                # number of training iterations
        self._setUpNetwork()
    def _setUpNetwork(self):
        """
        Our input is a 28 X 28 single row vector, so 784 rows
        Set up a the networks layers,
        generate random weights and bias for the given number of neurons
        :return:
        """

        # test setup for a standard competetive network, which involes a single layer
        # a competetive layer and the Kohonen rule

        #########################
        #   BOOK Example Weights

        ww1 = np.matrix([[.7071], [-.7071]])
        ww2 = np.matrix([[.7071], [.7071]])
        ww3 = np.matrix([[-1.0000], [0.0000]])
        W = np.concatenate( ( ww1.T,ww2.T,ww3.T))
        self.layers.append({'weights':W,'transFunc':'Compet','bias':0})
        #say we have three classes and want to classify to three reigons that is three neurons per class
        #our S x R is 3 X how many input we have our R is a 2x1 so S X R = 3 X 2
        #
        #
        #
        #
        #

        #per each neuron will have its own weights since our input is a 2d we have each neuron match the input
        # space by planting random weights in the input space
        weight_1 = np.matrix(np.random.uniform(0,1,size=(2,1)))
        weight_2 = np.matrix(np.random.uniform(0,1,size=(2,1)))
        weight_3 = np.matrix(np.random.uniform(0,1,size=(2,1)))

        #make our total weight Vector
        # W = {W1.T,
        #      W2.T,
        #      W3.T]
        _WeightVector = np.concatenate( (weight_1.T,weight_2.T,weight_3.T) )

        self.layers.append( {'weights':_WeightVector, 'transFunc':'Compet','bias':0}) # append the first layer, if more then

        #continue to append
        #self.plotWeights()
    def train(self , inputVectors):
        # give me a set of input vectors and for each input i will train the network
        #training involves

        #for each inputVector
        for p in inputVectors:
            n = np.dot(self.layers[0]['weights'], p)
            a = CompetitiveTrans(n) # got a
            # now get the index of the winning layer
            i_star = np.argmax(a)

            self.updateWeight(i_star,p)

    def plotWeights1(self):
        #plot the weights
        x = []
        y = []
        #nested matrix
        for i in self.layers[0]['weights']:
            x.append(i[0][0,0])
            y.append(i[0][0,1])
        plt.scatter(x,y,color='red', label='Weight Final')
        print("the Weights are:\n %s" % self.layers[0]['weights'])
        plt.axhline(linewidth=2, color='black')
        plt.axvline(linewidth=2, color='black')
        plt.legend()
        plt.grid()

    def plotResults(self):
        f = plt.figure(1)
        p1 = np.matrix([[-.1961], [.9806]])
        p2 = np.matrix([[.1961], [.9806]])
        p3 = np.matrix([[.9806], [.1961]])
        p4 = np.matrix([[.9806], [-.1961]])
        p5 = np.matrix([[-.5812], [-.8137]])
        p6 = np.matrix([[-.8137], [-.5812]])
        p = [p1, p2, p3, p4, p5, p6]
        x = []
        y = []
        for i in p:
            x.append(i[0][0, 0])
            y.append(i[1][0, 0])

        plt.scatter(x,y,color='blue',label='Input Patterns')


        #plot the weights
        x = []
        y = []
        #nested matrix
        for i in self.layers[0]['weights']:
            x.append(i[0][0,0])
            y.append(i[0][0,1])
        plt.scatter(x,y,color='red', label='Weight Final')
        print("the Weights are:\n %s" % self.layers[0]['weights'])
        plt.axhline(linewidth=2, color='black')
        plt.axvline(linewidth=2, color='black')
        plt.legend()
        plt.grid()

    def updateWeight(self, winningNeuron, inputVector):
        #update the weight of the winning neuron
        new_weights = self.layers[0]['weights'][winningNeuron].T + ( self.learning_rate * ( inputVector - self.layers[0]['weights'][winningNeuron].T))
        self.layers[0]['weights'][winningNeuron] = new_weights.T



        print("plot updated weights")
        #time.sleep(5)

        #f = plt.figure(3)
        #plot the weights
        x = []
        y = []
        #nested matrix
        for i in self.layers[0]['weights']:
            x.append(i[0][0,0])
            y.append(i[0][0,1])
        plt.scatter(x,y,color='red', label='Weight Final')
        plt.pause(5)
        print("the Weights are:\n %s" % self.layers[0]['weights'])

        plt.legend()

        plt.draw()

        #self.plotWeights() #
        #plt.show()


    def plotWeights(self, weights = None):
        #Plot the weights
        if weights is None:
            weights = self.layers[0]['weights'] # get the inital layer 1 weights

        x = []
        y = []
        #nested matrix
        for i in weights:
            x.append(i[0][0,0])
            y.append(i[0][0,1])
        print("the Weights are:\n %s" % weights)
        plt.axhline(linewidth=2, color='black')
        plt.axvline(linewidth=2, color='black')
        plt.grid()
        #plt.scatter(x,y, color='green')


        ##########################################################################
        #           PLOTTING THE INPUT VECTORS FOR TESTING PURPOSES
        # plot the inital input vectors to see if these weigths cover the input space
        p1 = np.matrix([[-.1961], [.9806]])
        p2 = np.matrix([[.1961], [.9806]])
        p3 = np.matrix([[.9806], [.1961]])
        p4 = np.matrix([[.9806], [-.1961]])
        p5 = np.matrix([[-.5812], [-.8137]])
        p6 = np.matrix([[-.8137], [-.5812]])
        p = [p1, p2, p3, p4, p5, p6]
        x = []
        y = []
        for i in p:
            x.append(i[0][0, 0])
            y.append(i[1][0, 0])

        plt.scatter(x,y,color='red',label='Input Patterns')
        plt.legend()

def preprocessData():
    # we need to rescale the data suince it is between [0,255]
    # our data is not normally distrbuted we do not have a mean or an STD
    # so dividing by the max 255 will bring our values between [0, 1]
    return -1
def pickleData():
    # pickle the data
    # faster load times
    mndata = MNIST('./data')

    images, labels = mndata.load_training()# loads only the training data
    ################################
    #pickle the data
    imageFile = open('./Sources/images.pickle','wb')
    pickle.dump(images,imageFile)
    labelsFile = open('./Sources/labels.pickle','wb')
    pickle.dump(labels, labelsFile)

    imageFile.close()
    labelsFile.close()
    #############################
def loadPickleData():
    imageVFile = open('./Sources/images.pickle','rb')
    imageVector = pickle.load(imageVFile)
    imageLFile = open('./Sources/labels.pickle','rb')
    labelVector = pickle.load(imageLFile)

    return imageVector, labelVector
def generateImageVector(imageV, labelV):
    fig = plt.figure(1)
    for i in range(1,10):
        plt.subplot(3,3,i)                      # 3 rows 3 columns, and subplot number starting at 1
        randomImage = random.randint(0,len(imageV))
        pixels = np.array(imageV[randomImage])  # pick a random image vector to display
        pixels = pixels.reshape((28,28))        # reshape the single row vector 784 X 1 to a square matrix
        plt.imshow(pixels, cmap='gray')         # put the image on the subplot
        plt.title(labelV[randomImage])          # show the label of the image, using the image vector
    plt.tight_layout()                          # prevent label overlays
def loadData():
    # check if the picked files exist, load them if so, else make them
    if os.path.isfile('./Sources/images.pickle') \
            and os.path.isfile('./Sources/labels.pickle'):
        imagesVector, labelsVector = loadPickleData()
        print("file exist")
        return imagesVector,labelsVector
    else:
        pickleData()
        print("does not exist")


def CompetitiveTrans(data):
    r = np.zeros_like(data)
    max = np.argmax(data) # get the index with the maximum value
    r[max] = 1.0
    return r # returns the maximum value the winning nurons index

def main():
    imagesVector, labelsVector = loadData() # load the data
    # print("Labels for the perspective index's %s" % labelsVector)
    # generateImageVector(imagesVector, labelsVector)
    #
    #
    #
    # f2 = plt.figure(2)
    # plt.subplot(2,1,1)
    # plt.hist(imagesVector[55])
    # plt.subplot(2,1,2)
    # p = np.array(imagesVector[55]).reshape((28,28))/255
    # print(p)
    # plt.imshow(p)
    #

    net = Network(1)
    #training patterns

    p1 = np.matrix([[-.1961], [.9806]])
    p2 = np.matrix([[.1961], [.9806]])
    p3 = np.matrix([[.9806], [.1961]])
    p4 = np.matrix([[.9806], [-.1961]])
    p5 = np.matrix([[-.5812], [-.8137]])
    p6 = np.matrix([[-.8137], [-.5812]])
    #p vector
    p = [p1, p2, p3, p4, p5, p6]

    net.train(p)


    print(imagesVector[5])

    #x = np.matrix(imagesVector[5]).reshape( (28,28))
    #plt.scatter(x)

    #plt.show()
if __name__ == "__main__":
    main()









