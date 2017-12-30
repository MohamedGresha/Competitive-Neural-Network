
from mnist import MNIST
import numpy as np
import pickle, os, random, time
from matplotlib import pyplot as plt
from neurolab.trans import Competitive, PureLin

np.random.seed(5)


class Network:
    def __init__(self, epoach=5, learning_rate=.02, verbose=False):
        """
        Default constructor
        """
        self.classes = 10
        self.subclasses = 50

        self.layers = []  # each index is a layer
        self.learning_rate = learning_rate  # the learning tate of the network
        self.epoach = epoach  # number of training iterations
        self._setUpNetwork()
        self.error_results = list()

        self.epoach_error = []
        self.epoach_accuracy = []

        self.makeFinalClassLayer(self.classes, self.subclasses)

    def _getHotEncoding(self, targetInteger):
        """
        One hot encoding returns a one got encoded value to the
        respective target Integer
        :param targetInteger:  ex 5 will return a 1x10
        ie: 5 -> [0,0,0,0,0,1,0,0,0,0,0]
        :return:  returns a  numpy matrix
        """
        _hot_encoding = np.matrix([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        return _hot_encoding[targetInteger]

    def makeFinalClassLayer(self, numClasses, numSubClasses):
        """
        For the final output layer we need a layer to combine all subclasses from the pervious layers to our
        final layer, therefore this method will create a one-hot encoding final layer for the number of classes/categories
        of the problem we are trying to solve.
        :return:
        """
        assert numSubClasses % numClasses == 0, "Error there is an unbalanced subclass count please recheck your settings"
        # make the one hot encoded matrix
        hot_encode = np.zeros(shape=(numClasses, numClasses))
        for i in range(len(hot_encode)):
            hot_encode[i][i] = 1
        # we know that we have 0 - 9 digits
        # this hot_encode hold for each index a unique value
        # each index is also associated with each digit starting from 0 -> 9
        # print(hot_encode)

        subclassPerClass = numSubClasses / numClasses  # get the number of sub classes per class in final layer

        # assign groups of subclasses to classes
        hot = 0
        m = np.matrix(np.zeros(shape=(numClasses, numSubClasses)))  # make a default subclass

        # print(np.matrix(hot_encode[0]).shape)
        # print("-----")
        for i in range(numSubClasses):
            # m.append(hot_encode[hot])
            if not i == 0 and i % subclassPerClass == 0:
                hot += 1

            m[:, i] = np.matrix(hot_encode[hot]).T
        # m = np.matrix(m)
        #
        # for i in range(0,50,5):
        #     print("class %s" % i)
        #     print("Grouping")
        #     print(m[:,i:i+5])
        # print(type(m))
        assert m.shape == (self.classes, self.subclasses)
        return m

    def _setUpNetwork(self):
        """
        Our input is a 28 X 28 single row vector, so 784 rows
        Set up a the networks layers,
        generate random weights and bias for the given number of neurons
        :return:
        """

        # so initally we want 50 subclasses so 50 neurons amongst  784 features

        # s is 50 R is 784

        W_1_0 = np.matrix(np.random.uniform(low=0, high=1, size=(
        50, 784)))  # 50 X 784#the input ares normizaed between 0 and 255 or 0,1

        assert W_1_0.shape == (
        50, 784), "Error initalizing weights "  # assert check that these parameters or checkpoints are forced.

        layer1 = {'weights': W_1_0.T,  # per following the equations Transpose such that each colum is a nueron
                  'transFunc': 'Compet',
                  'bias': 0}
        W_2_0 = self.makeFinalClassLayer(10, 50)

        assert W_2_0.shape == (self.classes, self.subclasses)
        layer2 = {'weights': W_2_0,
                  'transFunc': 'Purlin',
                  'bias': 0}

        # popualte the layers
        self.layers.append(layer1)
        self.layers.append(layer2)


        # print(W_1_0.shape)
        # print(W_2_0.shape)
        # continue to append
        # self.plotWeights()

    def train(self, inputVectors, targetVectors):
        """

        :param inputVectors: Given the input and target vectors
        :return:
        """
        # initialize the error array each index repersents an epoach
        # a iterating of the training
        self.numSamples = len(inputVectors)
        for iter in range(self.epoach):  # number of epoachs
            c_e = 0
            for pInput in range(len(inputVectors)):
                # grab the input training pattern and the target vector
                _P = np.matrix(np.divide(inputVectors[pInput], 255))  # get the input vector
                assert _P.shape == (1, 784), "Input vector dimensions error"
                targetIndex = targetVectors[pInput]
                _T = self._getHotEncoding(targetIndex).T

                # right now we exptect Weights to be 50X784
                assert self.layers[0]['weights'].shape == (784, 50), "Weights are correct dimensions"
                # shift back weights scuh that each row is a neurons prototype vector, the P_ input is a 1X784 which is fine
                netInput_1 = -abs(np.matrix(np.linalg.norm(self.layers[0]['weights'].T - _P,
                                                           axis=1)))  # get netinput #usually we need to transpose

                # but since the input vectors are already 1X50 row vectors we dont need to transpose
                # n = -abs(n)
                netInput_1 = netInput_1.T  # transpose the resules
                # print(n)


                # now propagate this input to the next layer
                # print("Actual output %s " % _t)

                a_1 = CompetitiveTrans(netInput_1)  # award the winning neruon

                #################### second layer



                assert self.layers[1]['weights'].shape == (self.classes, self.subclasses), "Error with layer 2 weights"

                assert a_1.shape == (self.subclasses, 1), "First layer shape failed"

                netInput_2 = np.dot(self.layers[1]['weights'], a_1)
                assert netInput_2.shape == (10, 1), "netinput 2 failed shape"

                pur = PureLin()
                a_2 = pur(netInput_2)  # stand them up #10X 1

                # tells us which class this input belongs too

                assert _T.shape == (10, 1), "Error with Target vector"
                # calculate the error

                mse = (np.square(_T - a_2)).mean()

                c_e += mse
                # print("MSE %s" % mse)

                # when weights are correctly classifiedmove the single winner to the input,
                # when weights are incorrectly classified move
                ########### update weights

                winning_neuron = np.argmax(a_1)  # wining neurons index or losing
                Copy_N = netInput_1.copy()  # copy the netinput from layer 1, which has all the distances of vectors

                # and their weights
                # that is a 50 X 1
                # now copy it to make sure we do not alter it, even if we did however it wouldnt' matter
                Copy_N[winning_neuron] = -np.inf  # sort it we need to find a way to get the runner up,
                # sort, get the winning neuron which is at the bottom, find the largest index and minus 1 to get the
                # make the above the least to win now find the second higest
                # second highest neuron


                runner_up = np.argmax(Copy_N)  # got the runner up
                # print("Winner:%s Runner Up %s " % (netInput_1[winning_neuron],netInput_1[runner_up]))
                assert self.layers[0]['weights'].shape == (784, 50), "ERROR with updaing weights"

                if mse == 0.0:
                    # correctly classified
                    self._updateWeights(_P, winning_neuron)
                else:
                    # false positive, update the runner up
                    self._updateWeights(_P, winning_neuron, runner_up)

                if mse == 0.0:  # correct classified
                    # c_e +=1
                    # access rows in a matrix Matrix[i,:], where i is the index of the entire row you want to grab
                    # access columns in a matrix Matrix[:,i], where i is the index of the coulmn you want
                    # print("Input pattern %s " % np.matrix(_P).T)

                    # print("Choosen Weight vector W%s is %s " % (winning_neuron, self.layers[0]['weights'][:,winning_neuron]))

                    assert self.layers[0]['weights'][:, winning_neuron].shape == (784, 1), "ERROR with choosen one"

                    assert _P.T.shape == (784, 1)
                    self.layers[0]['weights'][:, winning_neuron] = self.layers[0]['weights'][:, winning_neuron] + (
                    self.learning_rate * (_P.T - self.layers[0]['weights'][:, winning_neuron]))

                    # print('UPDATEED WEIGHTS: %s' % new_weights)
                else:

                    # print("Push this neuron away : %s" % np.argmax(a_1))
                    # print(a_1)
                    # new_weights = self.layers[0]['weights'][:, winning_neuron] - (
                    # self.learning_rate * (_P.T - self.layers[0]['weights'][:, winning_neuron]))
                    self.layers[0]['weights'][:, winning_neuron] = self.layers[0]['weights'][:, winning_neuron] - (
                    self.learning_rate * (_P.T - self.layers[0]['weights'][:, winning_neuron]))

                    # bring runner up closer
                    self.layers[0]['weights'][:, runner_up] = self.layers[0]['weights'][:, runner_up] + (
                    self.learning_rate * (_P.T - self.layers[0]['weights'][:, runner_up]))

            self.epoach_error.append(1 - (c_e / len(inputVectors)))

            self.epoach_accuracy.append((c_e / len(inputVectors)))

            # self.epoach_error.append( abs( (c_e )/ len(inputVectors)))
            print("ERROR FOR epoch: %s | Current Error Rate: %s |current error %s, numVector %s" % (
            iter, self.epoach_error[iter], c_e, len(inputVectors)))
            ################################### WEIGHT UPDATE   ########################################

    def _updateWeights(self, inputVector, winningNeuron, runnerup=None):
        """
        If you give me a runner up that means that there is a WRONG classification
        :param winningNeuron: Int of the winning or false winning neron
        :param runnerup: the runner up if we have a false positive
        :return:
        """
        moveCloser = winningNeuron  # default
        if runnerup is not None:
            # we had a wrong classification move this neuron closer , which is the runner up neron
            moveCloser = runnerup
            # move the false positive neuron away
            self.layers[0]['weights'][:, winningNeuron] = self.layers[0]['weights'][:, winningNeuron] - (
            self.learning_rate * (inputVector.T - self.layers[0]['weights'][:, winningNeuron]))

        # move Closer adjust if we have a runner up then move closer i the runner up if we do not we default the moveCloser as winning neuron
        # move the runner up closer and the winner farter, we always move something closer
        self.layers[0]['weights'][:, moveCloser] = self.layers[0]['weights'][:, moveCloser] + (
        self.learning_rate * (inputVector.T - self.layers[0]['weights'][:, moveCloser]))

    def plotPerformance(self):

        ff = plt.figure(8)
        print("Error: %s" % self.epoach_error)
        print("Accuracy: %s " % self.epoach_accuracy)
        x = np.arange(1, self.epoach + 1)
        # for i in range(len(self.error_results)):
        #     self.error_results[i] = self.error_results[i][0,0]
        #
        # plt.plot(self.error_results)
        plt.title(r"Network Mean Square Error (lower is better) $\alpha$ %.4f  $\eta$ = %s" % (
        self.learning_rate, self.numSamples))
        plt.plot(x, self.epoach_error, color='m', label="network error")
        plt.xlabel("Epochs iteration")
        plt.ylabel("Error mean %")
        plt.tight_layout()
        plt.legend()
        fff = plt.figure(23)
        plt.title(r"Network Accuracy $\alpha$ %.4f $\eta$ = %s" % (self.learning_rate, self.numSamples))
        plt.xlabel("Epochs Iteration")
        plt.plot(x, self.epoach_accuracy, color='r', label="network accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

        #
        # fig = plt.figure(69696969)
        #
        # plt.xticks(np.arange(1,self.epoach+1))
        # plt.title(r"Network Error and Accuracy $\alpha$%s.f $\eta$=%s" %(self.learning_rate,self.numSamples))
        # plt.xlabel("Epoch Iterations")
        # plt.ylabel("Error percent")
        # plt.plot(np.arange(1, self.epoach + 1), self.epoach_error, color='r', label="Error Rate")
        # plt.plot(np.arange(1, self.epoach + 1), self.epoach_accuracy, color='b', label="Accuracy Rate")
        # #plt.xlim([0, 1])
        # plt.ylim([0, 1])
        # plt.legend()
        # plt.show()

    def predict(self, inputVector, targetVector):
        """
        Test the network's performance to perdict what the given handwritten vector is
        :param inputVector:
        :param targetVector:
        :return:
        """
        _t = np.matrix(self._getHotEncoding(targetVector)).T  # returns the target vector with hot encoding
        # print("Encoding %s -> %s" % (targetVector, _t))
        # print("GOT : %s -> hot encoded to: %s" % ( targetVector, _t))
        _p = np.matrix(np.divide(np.matrix(inputVector), 255))
        assert _p.shape == (1, 784)
        assert self.layers[0]['weights'].shape == (784, 50)
        assert _t.shape == (10, 1)
        n_1 = -abs(np.matrix((np.linalg.norm(self.layers[0]['weights'].T - _p, axis=1))))
        a_1 = CompetitiveTrans(n_1.T)
        assert a_1.shape == (50, 1)

        # second layer
        assert self.layers[1]['weights'].shape == (10, 50)
        n_2 = np.dot(self.layers[1]['weights'], a_1)
        assert n_2.shape == (10, 1)
        pur = PureLin()
        a_2 = pur(n_2)  # stand them up #10X 1
        assert a_2.shape == (10, 1)
        # print("exptected: %s |\n Got: %s" % (_t.T, a_2.T))

        return np.array_equal(_t, a_2)


def pickleData():
    # pickle the data
    # faster load times
    mndata = MNIST('./data')

    images, labels = mndata.load_training()  # loads only the training data
    ################################
    # pickle the data
    imageFile = open('./Sources/images.pickle', 'wb')
    pickle.dump(images, imageFile)
    labelsFile = open('./Sources/labels.pickle', 'wb')
    pickle.dump(labels, labelsFile)
    imageFile.close()
    labelsFile.close()

    ################## Testing


    # images, labels = mndata.load_training()# loads only the training data
    images, labels = mndata.load_testing()
    imageFile = open('./Sources/images_Test.pickle', 'wb')
    pickle.dump(images, imageFile)
    labelsFile = open('./Sources/labels_Test.pickle', 'wb')
    pickle.dump(labels, labelsFile)

    imageFile.close()
    labelsFile.close()
    #############################


def loadPickleData():
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

    return imageVector, labelVector, imageVector_Test, labelVector_Test


def generateImageVector(imageV, labelV):
    fig = plt.figure(1)
    for i in range(1, 10):
        plt.subplot(3, 3, i)  # 3 rows 3 columns, and subplot number starting at 1
        randomImage = random.randint(0, len(imageV))
        pixels = np.array(imageV[randomImage])  # pick a random image vector to display
        pixels = pixels.reshape((28, 28))  # reshape the single row vector 784 X 1 to a square matrix
        plt.imshow(pixels, cmap='gray')  # put the image on the subplot
        plt.title(labelV[randomImage])  # show the label of the image, using the image vector
    plt.tight_layout()  # prevent label overlays


def loadData():
    # check if the picked files exist, load them if so, else make them
    if os.path.isfile('./Sources/images.pickle') \
            and os.path.isfile('./Sources/labels.pickle'):
        imagesVector, labelsVector, imageV_test, lablV_test = loadPickleData()
        print("file exist")
        return imagesVector, labelsVector, imageV_test, lablV_test
    else:
        pickleData()
        print("does not exist")


def testLVQNetwork():
    # make an instasnces of LVQNetwork
    # take the testing data,
    # run the test using the 'predict
    # function to test the network against what it gives and what wee expect


    imagesVector, labelsVector, imageVector_Test, labelsVector_Test = loadData()  # load the data

    generateImageVector(imageVector_Test, labelsVector_Test)

    f = open('./saveNetwork.pickle', 'rb')

    network = pickle.load(f)
    f.close()

    results = dict()
    # initalize dict
    for i in range(10):
        # 0,...,9 inclusive
        results[i] = {'testCount': 0,
                      'correctCount': 0,
                      'classifiedErrors': list()}

    for pattern in range(len(imageVector_Test)):
        # for each testing pattern's index
        result = network.predict(imageVector_Test[pattern], labelsVector_Test[pattern])
        # true if they are classified correctly false otherwise
        if result == True:
            # corectly classified
            results[labelsVector_Test[pattern]]['correctCount'] += 1  # increment
        else:
            results[labelsVector_Test[pattern]]['classifiedErrors'].append(labelsVector_Test[
                                                                               pattern])  # record the patterns that were miss classified maybe get some insight on whats going on
        results[labelsVector_Test[pattern]]['testCount'] += 1  # increment number of times we've tested this pattern
        # print(result)
    print("TOTAL TEST: %s" % len(imageVector_Test))
    # print the results in a

    x = np.arange(0, 10)
    y = list()
    y_total = list()
    for i in sorted(results):
        y.append(results[i]['correctCount'])
        y_total.append(results[i]['testCount'])
    # print(y)
    # print(results)

    ff = plt.figure(6969)

    plt.bar(x + .25, y_total, label="Total Seen", color='gray')
    plt.bar(x, y, label="Num Correctly Classified")
    # plt.subplot(1,1,1)

    plt.xticks(x, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    plt.ylabel("Number of correct classified")
    # plt.xlabel("Digit")
    plt.title("Performance for Number of Correctly Classified")
    plt.legend()

    plt.show()


def CompetitiveTrans(data):
    r = np.zeros_like(data)
    max = np.argmax(data)  # get the index with the maximum value
    r[max] = 1.0
    return r  # returns the maximum value the winning nurons index


def convertIntToTargetEncoded(number):
    imageLFile = open('./Sources/labels.pickle', 'rb')
    labelVector = pickle.load(imageLFile)
    db = np.array([
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    return db[number]


def main():
    TRAINING = False

    if TRAINING:
        imagesVector, labelsVector, imageVector_Test, labelsVector_Test = loadData()  # load the data
        net = Network(epoach = 12, learning_rate = .005)

        num_samples = 600
        net.train(imagesVector[0:num_samples], labelsVector[0:num_samples])
        f = open('./saveNetwork.pickle', 'wb')
        pickle.dump(net, f)
        f.close()
        net.predict(imagesVector[3000], labelsVector[3000])
        net.plotPerformance()
    else:

        testLVQNetwork()


if __name__ == "__main__":
    main()
