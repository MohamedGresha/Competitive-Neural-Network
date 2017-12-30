# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
# =#| Author: Danny Ly MugenKlaus|RedKlouds
# =#| File:   CompetitiveNetwork.py
# =#| Date:   12/6/2017
# =#|
# =#| Program Desc:
# =#|
# =#| Usage:
# =#|
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|

from mnist import MNIST
import numpy as np
import pickle, os, random
from matplotlib import pyplot as plt
from neurolab.trans import PureLin, LogSig

np.random.seed(5)


class Network:
    def __init__(self, epoch=5, learning_rate=.1, verbose=False, dataDrivenInitalization=True, conscience=False):
        """

        This is a competitive Artificial Neural Network
        using the Learning Vector Quantization method
        Discovered by Kohonen
        Default constructor
        """

        # Testing is 50 subclasses,and 10 final classes
        # with input as a 1 X 784 input vector
        self.classes = 10
        self.subclasses = 5000
        self.inputVectorSize = 784
        self.useDDI = dataDrivenInitalization
        self.conscience = conscience

        self.layers = []  # each index is a layer
        self.learning_rate = learning_rate  # the learning tate of the network
        self.epoch = epoch  # number of training iterations

        self._setUpNetwork()

        self.error_results = list()

        self.epoch_error = []
        self.epoch_accuracy = []

        self._makeFinalClassLayer(self.classes, self.subclasses)

    def _dataDrivenInitalization(self, subclasses, classes, imageVector, labelVector):

        if subclasses % classes != 0:
            raise Exception("Uneven subclass to classes please recheck parameters")
        elif subclasses < 0 or classes < 0:
            raise Exception("Cannot have negative neurons")
        elif len(imageVector) < 0 or len(labelVector) < 0:
            raise Exception("Empty input and target vectors")
        neuronsPerClass = int(subclasses / classes)
        # each neuron in the group should be a random input vector of the input

        data_X = np.matrix(imageVector)
        data_Y = np.array(labelVector)

        # make a default zero matrix to manipulate

        d = np.matrix(np.zeros( (subclasses, data_X[0].size) ) )
        print(len(imageVector[0]))
        print(imageVector[0])
        #d = np.matrix(np.zeros((subclasses, self.inputVectorSize)))

        # for the rows 0 -> 4 should be 0 digit 0
        digit = 0
        for i in range(0, subclasses, neuronsPerClass):
            ve = data_X[data_Y == digit]
            for j in range(i, i + neuronsPerClass):
                ridx = random.randint(0, len(ve))
                print(d[j,:])
                d[j, :] = ve[ridx]
            digit += 1

        #assert d.shape == (self.subclasses, self.inputVectorSize)

        self.deadDetection = d.copy()
        ff = open("./CompetitiveNetworkStore/detectDead.p", 'wb')
        pickle.dump(self.deadDetection, ff)
        ff.close()

        self.layers[0]['weights'] = d.T

        print("finished Data Driven Initialization")
    def _getTransFunction(self, transFunc):
        key = transFunc.lower()
        case = \
            {
            'PURELIN': PureLin(),
            'LOGSIG' : LogSig(),
            'COMPET' : self._competitiveTrans()
            }
        return case[key]
    def _getHotEncoding(self, targetInteger):
        """
        One hot encoding returns a one got encoded value to the
        respective target Integer
        :param targetInteger:  ex 5 will return a 1x10
        ie: 5 -> [0,0,0,0,0,1,0,0,0,0,0]
        :return:  returns a  numpy matrix
        """
        if targetInteger < 0 or targetInteger > 9:
            return False
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

    def _makeFinalClassLayer(self, numClasses, numSubClasses):
        """
        For the final output layer we need a layer to combine all subclasses from the pervious layers to our
        final layer, therefore this method will create a one-hot encoding final layer for the number of classes/categories
        of the problem we are trying to solve.
        :return: np matrix classes x subclasses
        """
        if numClasses < 0 or numSubClasses < 0:
            return False
        if numSubClasses % numClasses != 0:
            raise Exception("unequal hidden layer to final layer mapping")
        #assert numSubClasses % numClasses == 0, "Error there is an unbalanced subclass count please recheck your settings"
        # make the one hot encoded matrix
        hot_encode = np.zeros(shape=(numClasses, numClasses))

        for i in range(len(hot_encode)):
            hot_encode[i][i] = 1
        # we know that we have 0 - 9 digits
        # this hot_encode hold for each index a unique value
        # each index is also associated with each digit starting from 0 -> 9

        subclassPerClass = numSubClasses / numClasses  # get the number of sub classes per class in final layer

        # assign groups of subclasses to classes
        hot = 0
        m = np.matrix(np.zeros(shape=(numClasses, numSubClasses)))  # make a default subclass

        for i in range(numSubClasses):
            # m.append(hot_encode[hot])
            if not i == 0 and i % subclassPerClass == 0:
                hot += 1

            m[:, i] = np.matrix(hot_encode[hot]).T

        #assert m.shape == (self.classes, self.subclasses)
        return m

    def _competitiveTrans(self, data):
        r = np.zeros_like(data)
        max = np.argmax(data)  # get the index with the maximum value
        r[max] = 1.0
        return r  # returns the maximum value the winning nurons index

    def _prelinTrans(self, data):
        p = PureLin()
        return p(data)

    def _setUpNetwork(self):
        """
        Our input is a 28 X 28 single row vector, so 784 rows
        Set up a the networks layers,
        generate random weights and bias for the given number of neurons
        :return:
        """

        # s is 50 R is 784
        # np.random.seed(0)
        W_1_0 = np.matrix(np.random.uniform(low=0, high=1, size=(
            self.subclasses, self.inputVectorSize)))  # 50 X 784#the input ares normalized between 0 and 255 or 0,1


        print(W_1_0.shape)
        print(W_1_0.T.shape)

        assert W_1_0.shape == (
            self.subclasses,
            self.inputVectorSize), "Error initalizing weights "  # assert check that these parameters or checkpoints are forced.

        B_1_0 = np.matrix(np.random.uniform(0, .5, (1, self.subclasses)))
        assert B_1_0.shape == (1, self.subclasses)
        layer1 = {'weights': W_1_0.T,  # per following the equations Transpose such that each colum is a nueron
                  'transFunc': 'Compet',
                  'bias': B_1_0.T}

        W_2_0 = self._makeFinalClassLayer(self.classes, self.subclasses)

        assert W_2_0.shape == (self.classes, self.subclasses)

        layer2 = {'weights': W_2_0,
                  'transFunc': 'Purlin',
                  'bias': 0}

        # popualte the layers
        self.layers.append(layer1)
        self.layers.append(layer2)

    def train(self, inputVectors, targetVectors):
        """
        pulbic function to train the network
        :param inputVectors: Given a input matrix
        :parem targetVectors: Given a inputVectors
        :return: None, Network has been trained
        """

        self.numSamples = len(inputVectors)
        if self.useDDI:
            self._dataDrivenInitalization(self.subclasses, self.classes, inputVectors, targetVectors)

        decay_rate = 10
        for iter in range(self.epoch):  # for each epoch
            c_e = 0  # initialize the current error for this epoch
            for pInput in range(len(inputVectors)):

                # Normalize the input vector by 255.0 ( normalize the input )
                # _P = np.matrix( np.divide(inputVectors[pInput], 255.0) )

                _P = np.matrix(inputVectors[pInput])

                # _P = (_P - _P.min())/ (_P.max() - _P.min())

                assert _P.shape == (1, self.inputVectorSize), "Input vector dimensions error"

                # get the target integer of the respective input
                targetIndex = targetVectors[pInput]
                # get the one-hot-encoded vector of the respective target input integer
                _T = self._getHotEncoding(targetIndex).T

                # right now we exact Weights to be 50X784
                assert self.layers[0]['weights'].shape == (
                self.inputVectorSize, self.subclasses), "Weights are correct dimensions"

                # calculate the net input of the first competitive layer
                # using ecludiean distance of two vectors
                # -|| dist ||
                netInput_1 = -abs(np.matrix(np.linalg.norm(self.layers[0]['weights'].T - _P,
                                                           axis=1)))

                netInput_1 = netInput_1.T  # transpose the results

                ## conscience
                if self.conscience:
                    netInput_1 = netInput_1 + self.layers[0]['bias']
                ###########
                # calculate the output of the first layer
                a_1 = self._competitiveTrans(netInput_1)

                #################### second layer ######################



                assert self.layers[1]['weights'].shape == (self.classes, self.subclasses), "Error with layer 2 weights"

                assert a_1.shape == (self.subclasses, 1), "First layer shape failed"

                # net input of the second purline layer w^2 * a^1
                # this combines all subclasses to the respective classes in the output layer
                netInput_2 = np.dot(self.layers[1]['weights'], a_1)

                assert netInput_2.shape == (self.classes, 1), "netinput 2 failed shape"

                # calculate the output of the second layer,
                a_2 = self._prelinTrans(netInput_2)  # stand them up #10X 1

                # tells us which class this input belongs too

                assert _T.shape == (self.classes, 1), "Error with Target vector"

                # calculate the error

                mse = (np.square(_T - a_2)).mean()

                ########### update weights ###########

                # get the potentially winning neuron

                winning_neuron = np.argmax(a_1)

                # deep copy the the net input to modified
                Copy_N = netInput_1.copy()
                # now set the first winning to -inf
                Copy_N[winning_neuron] = -np.inf

                runner_up = np.argmax(Copy_N)  # Get the second winner up from the copied net input

                correct = np.array_equal(_T, a_2)
                assert self.layers[0]['weights'].shape == (
                self.inputVectorSize, self.subclasses), "ERROR with updaing weights"

                # if mse == 0.0:
                if correct:
                    c_e += 1
                    # correctly classified
                    self._updateWeights(_P, winning_neuron)
                else:
                    # false positive, update the runner up
                    self._updateWeights(_P, winning_neuron, runner_up)

            if iter > 1:
                # we have at least 2 iterations
                # check if the error is within a spcific distance
                improv = self.epoch_error[iter - 1] - round((1 - (c_e / len(inputVectors))), 7)
                print("Current improvment %s" % improv)
                if improv > 0.06:
                    self.learning_rate = round(self.learning_rate - .006, 4)
                    print("decreasing LR %s" % self.learning_rate)
                elif improv < 0.0006:
                    self.learning_rate = round(self.learning_rate + .002, 4)
                    print("Increasing LR %s " % self.learning_rate)

            self.epoch_error.append(round(1 - (c_e / len(inputVectors)), 7))
            self.epoch_accuracy.append(round((c_e / len(inputVectors)), 7))

            print(
                "ERROR FOR epoch: %s | Current Error Rate: %s | Numbers Correctly Classified: %s | Total Patterns seen %s" % (
                    iter, self.epoch_error[iter], c_e, len(inputVectors)))
            ################################### WEIGHT UPDATE   ########################################

    def _updateWeights(self, inputVector, winningNeuron, runnerup=None):
        """
        If you give me a runner up that means that there is a WRONG classification
        :param winningNeuron: Int of the winning or false winning neron
        :param runnerup: the runner up if we have a false positive
        :return: None, Weights 2 neurons will be updated if false positive, else single neuron is updated.
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

        #######################
        # conscience
        # winning neuron gets its bias reduced while all others are multipled by a scalar

        # save the bias for the old bias
        if self.conscience:
            hold = self.layers[0]['bias'][moveCloser]
            fill = np.full((self.subclasses, 1), .9)
            # multiply all bias by this scaler
            self.layers[0]['bias'] = np.multiply(self.layers[0]['bias'], fill)
            # penalize the winning neuron
            self.layers[0]['bias'][moveCloser] = hold - .2

    def findDeadNeuron(self):
        # weights currently are col are neurons
        count = 0
        for i in range(self.subclasses):
            if np.array_equal(self.deadDetection.T[:, i], self.layers[0]['weights'][:, i]):
                count += 1
        return count

    def plotPerformance(self):

        ff = plt.figure(8)
        print("Error: %s" % self.epoch_error)
        print("Accuracy: %s " % self.epoch_accuracy)
        print("Number of dead Nurons %s" % self.findDeadNeuron())
        x = np.arange(1, self.epoch + 1)

        #
        # ff = plt.figure(342)
        # plt.ylim([0.0,1.0])
        # plt.title(r"Network Error vs Accuracy $\alpha$ %.3f $\eta$=%s" % (self.learning_rate,self.numSamples))
        # plt.xlabel("Epochs Iteratons")
        # plt.ylabel("unit in percent(%)")
        # plt.plot(x, self.epoch_accuracy, color='r', label="network accuracy")
        # plt.plot(x, self.epoch_error, color='m', label="network error")
        # plt.legend()
        # plt.tight_layout()
        # plt.show()

        plt.title(r"Network Mean Square Error (lower is better) $\alpha$ %.4f  $\eta$ = %s" % (
            self.learning_rate, self.numSamples))
        plt.plot(x, self.epoch_error, color='m', label="network error")
        plt.xlabel("Epochs iteration")
        plt.ylabel("Error mean %")
        plt.tight_layout()
        plt.legend()
        fff = plt.figure(23)
        plt.title(r"Network Accuracy $\alpha$ %.4f $\eta$ = %s" % (self.learning_rate, self.numSamples))
        plt.xlabel("Epochs Iteration")
        plt.plot(x, self.epoch_accuracy, color='r', label="network accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

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

        # _p = np.matrix(np.divide(np.matrix(inputVector), 255.0))
        _p = np.matrix(inputVector)
        assert _p.shape == (1, self.inputVectorSize)
        assert self.layers[0]['weights'].shape == (self.inputVectorSize, self.subclasses)
        assert _t.shape == (self.classes, 1)
        n_1 = -abs(np.matrix((np.linalg.norm(self.layers[0]['weights'].T - _p, axis=1))))
        a_1 = self._competitiveTrans(n_1.T)
        assert a_1.shape == (self.subclasses, 1)

        # second layer
        assert self.layers[1]['weights'].shape == (self.classes, self.subclasses)
        n_2 = np.dot(self.layers[1]['weights'], a_1)
        assert n_2.shape == (self.classes, 1)
        pur = PureLin()
        a_2 = pur(n_2)  # stand them up # 10 X 1
        assert a_2.shape == (self.classes, 1)

        return np.array_equal(_t, a_2)

    def savePrototypeWeights(self):
        """
        Analytics save the current trained prototypes to a file to analyze
        :return:
        """

        import pickle
        f = open("./CompetitiveNetworkStore/NetworkWeights.pick", 'wb')
        pickle.dump(self.layers[0]['weights'].T, f)
        f.close()

        # def dataCollection(self, fileName):
        #     # collect data for inital weights
        #
        #     # save the state of the network save


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


def testLVQNetwork(numSamples, learningRate, epochs):
    # make an instasnces of LVQNetwork
    # take the testing data,
    # run the test using the 'predict
    # function to test the network against what it gives and what wee expect


    imagesVector, labelsVector, imageVector_Test, labelsVector_Test = loadData()  # load the data

    generateImageVector(imageVector_Test, labelsVector_Test)

    f = open('./CompetitiveNetworkStore/saveNetwork.pickle', 'rb')

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
    print(results)

    ff = plt.figure(6969)

    plt.bar(x + .25, y_total, label="Total Seen", color='gray')
    plt.bar(x, y, label="Num Correctly Classified")
    # plt.subplot(1,1,1)

    plt.xticks(x, ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9'))
    plt.ylabel("Number of correct classified")
    # plt.xlabel("Digit")
    plt.title(r"Network Perf. Correctly classifed $\alpha$%.2f$\eta$%sepochs%s" % (learningRate, numSamples, epochs))
    plt.legend()

    plt.show()


def main():
    TRAINING = True

    # ep = 15
    ep = 50
    lr = .08
    # lr = 7
    ns = 60000
    if TRAINING:

        imagesVector, labelsVector, imageVector_Test, labelsVector_Test = loadData()  # load the data
        net = Network(epoch=ep, learning_rate=lr, dataDrivenInitalization=True)
        # small leariong rate .006 made 9 inpossible to find
        num_samples = ns
        net.train(imagesVector[0:num_samples], labelsVector[0:num_samples])

        # save the state of the network for testing
        f = open('./CompetitiveNetworkStore/saveNetwork.pickle', 'wb')
        pickle.dump(net, f)
        f.close()

        net.savePrototypeWeights()

        net.findDeadNeuron()
        net.predict(imagesVector[3000], labelsVector[3000])
        net.plotPerformance()
    else:

        testLVQNetwork(ns, lr, ep)


if __name__ == "__main__":
    main()
