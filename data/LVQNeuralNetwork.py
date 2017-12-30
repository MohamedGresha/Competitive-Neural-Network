#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   LVQNeuralNetwork.py
#=#| Date:   12/3/2017
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|



from mnist import MNIST
import numpy as np
import pickle, os, random, time
from matplotlib import pyplot as plt
from neurolab.trans import Competitive, PureLin
#plt.ion()
plt.axhline(linewidth=2, color='black')
plt.axvline(linewidth=2, color='black')
plt.grid()
plt.xlim([-1,1])
plt.ylim([-1,1])
class Network:

    def __init__(self, epoach = 5, learning_rate= .02, verbose = False):
        """
        Default constructor
        """

        self.layers = []                    # each index is a layer
        self.learning_rate = learning_rate  # the learning tate of the network
        self.epoach = epoach                # number of training iterations
        self._setUpNetwork()
        self.error_results = list()
        self.targetStore = np.array([
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


        self.epoach_error = []
        self.epoach_accuracy = []
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
    def _setUpNetwork(self):
        """
        Our input is a 28 X 28 single row vector, so 784 rows
        Set up a the networks layers,
        generate random weights and bias for the given number of neurons
        :return:
        """

        #so initally we want 50 subclasses so 50 neurons amongst  784 features

        # s is 50 R is 784

        W_1_0 = np.matrix(np.random.uniform(low = -1, high = 1,size = (50,784))) # 50 X 784
        layer1 = {'weights': W_1_0.T,
                  'transFunc':'Compet',
                  'bias':0}
        W_2_0 = layer2Weight()
        layer2 = {'weights': W_2_0,
                  'transFunc':'Purlin',
                  'bias':0}

        #popualte the layers
        self.layers.append(layer1)
        self.layers.append(layer2)


        print(W_1_0.shape)
        print(W_2_0.shape)
        #continue to append
        #self.plotWeights()

    def train1(self, inputVectors, targetVectors):
        """

        :param inputVectors: Given the input and target vectors
        :return:
        """
        # initalize the error array each index repersents an epoach
        # a iterationg of the training

        for iter in range(self.epoach):#number of epoachs
            c_e = 0

            for pInput in range(len(inputVectors)):
                # grab the input training pattern and the target vector
                _P = np.matrix(np.divide(inputVectors[pInput], 255))  # get the input vector
                targetIndex = targetVectors[pInput]
                _t = self.targetStore[targetIndex]

                n = np.matrix( np.linalg.norm(self.layers[0]['weights'].T - _P, axis=1) )  # get netinput #usually we need to transpose
                # but since the input vectors are already 1X50 row vectors we dont need to transpose
                n = -abs(n)
                n = n.T
                # print(n)


                # now propogate this input to the next layer
                # print("Actual output %s " % _t)

                a_1 = CompetitiveTrans(n)
                # print(a_1)
                n_2 = np.dot(self.layers[1]['weights'], a_1)

                pur = PureLin()
                a_2 = pur(n_2.T)  # stand them up #10X 1
                # calculate the error
                error = abs(_t - a_2)
                # print("ERROR: %s" % error)
                #sta
                #
                # print("Final output:    %s " % n_2.T)
                # print("FINAL OUTPUT A:  %s" % a_2)
                # print("Expected Output: %s " % _t.T)

                e = np.dot(error, error.T)

                # self.error_results[pInput] += (np.square(self._cur_error))

                ###update the weights in the first layer by moving them closer or furthuer away
                # if e == 0:
                # correctly classified move the weights closer to the input vector
                # print("WINNING NEURON: %s " % a_1)
                # np.argmax(a_1)

                c_e += e[0][0,0]/len(inputVectors)
                #print("Current error %s" % c_e)

                # get the willint neuron

                winning_neuron = np.argmax(a_1)
                if e == 0:
                    # correct classified

                    # access rows in a matrix Matrix[i,:], where i is the index of the entire row you want to grab
                    # access columns in a matrix Matrix[:,i], where i is the index of the coulmn you want
                    # print("Input pattern %s " % np.matrix(_P).T)

                    # print("Choosen Weight vector W%s is %s " % (winning_neuron, self.layers[0]['weights'][:,winning_neuron]))
                    new_weights = self.layers[0]['weights'][:, winning_neuron] + (
                    self.learning_rate * (_P.T - self.layers[0]['weights'][:, winning_neuron]))
                    self.layers[0]['weights'][:, winning_neuron] = new_weights
                    # print('UPDATEED WEIGHTS: %s' % new_weights)
                else:

                    # print("Push this neuron away : %s" % np.argmax(a_1))
                    # print(a_1)
                    new_weights = self.layers[0]['weights'][:, winning_neuron] - (
                    self.learning_rate * (_P.T - self.layers[0]['weights'][:, winning_neuron]))
                    self.layers[0]['weights'][:, winning_neuron] = new_weights

            self.epoach_accuracy.append( abs( 1 - ( c_e / len(inputVectors) ) ) )
            #self.epoach_error.append((c_e / len(inputVectors) ) )
            self.epoach_error.append( abs( (c_e )/ len(inputVectors)))
            print("ERROR FOR iter %s:  %s current error %s, numVector %s" % (iter,self.epoach_error[iter], c_e, len(inputVectors)))
                    ################################### WEIGHT UPDATE   ########################################

    def train(self , inputVectors, targetVectors):
        """

        :param inputVectors: Given the input and target vectors
        :return:
        """
        #initalize the error array each index repersents an epoach
        # a iterationg of the training
        self.error_results = [0] * len(inputVectors) # number of training samples


        for pInput in range(len(inputVectors)):
            # grab the input training pattern and the target vector
            _P = np.matrix(np.divide(inputVectors[pInput],255)) # get the input vector
            targetIndex = targetVectors[pInput]
            _t = self.targetStore[targetIndex]
            print(_P.shape)
            n = np.matrix(np.linalg.norm(self.layers[0]['weights'].T - _P, axis =1)) # get netinput #usually we need to transpose
            # but since the input vectors are already 1X50 row vectors we dont need to transpose
            n = -abs( n )
            n = n.T
            #print(n)


            #now propogate this input to the next layer
            #print("Actual output %s " % _t)

            a_1 = CompetitiveTrans(n)
            #print(a_1)
            n_2 = np.dot(self.layers[1]['weights'], a_1)


            pur = PureLin()
            a_2 = pur(n_2.T) # stand them up #10X 1
            #calculate the error
            error = _t - a_2
            # print("ERROR: %s" % error)
            #
            #
            # print("Final output:    %s " % n_2.T)
            # print("FINAL OUTPUT A:  %s" % a_2)
            # print("Expected Output: %s " % _t.T)
            self.cur_error = error
            e = np.dot(error,error.T)
            print("ERROR : %s %s" % (e == 2, e))
            #self.error_results[pInput] = np.dot(error.T,error)
            #self.error_results[pInput] += (np.square(self._cur_error))

            ###update the weights in the first layer by moving them closer or furthuer away
            #if e == 0:
                # correctly classified move the weights closer to the input vector
            #print("WINNING NEURON: %s " % a_1)
            #np.argmax(a_1)


            #get the willint neuron

            winning_neuron = np.argmax(a_1)
            if e == 0:
                #correct classified
                print(a_1)
#access rows in a matrix Matrix[i,:], where i is the index of the entire row you want to grab
#access columns in a matrix Matrix[:,i], where i is the index of the coulmn you want
                #print("Input pattern %s " % np.matrix(_P).T)

                #print("Choosen Weight vector W%s is %s " % (winning_neuron, self.layers[0]['weights'][:,winning_neuron]))
                new_weights = self.layers[0]['weights'][:,winning_neuron] + ( self.learning_rate * ( _P.T - self.layers[0]['weights'][:,winning_neuron]) )
                self.layers[0]['weights'][:,winning_neuron] = new_weights
                #print('UPDATEED WEIGHTS: %s' % new_weights)
            else:

                #print("Push this neuron away : %s" % np.argmax(a_1))
                #print(a_1)
                new_weights = self.layers[0]['weights'][:,winning_neuron] - ( self.learning_rate * (_P.T - self.layers[0]['weights'][:,winning_neuron]) )
                self.layers[0]['weights'][:,winning_neuron] = new_weights


            ################################### WEIGHT UPDATE   ########################################

    def updateWeight(self, winningNeuron, inputVector):
        #update the weight of the winning neuron
        new_weights = self.layers[0]['weights'][winningNeuron].T + ( self.learning_rate * ( inputVector - self.layers[0]['weights'][winningNeuron].T))
        self.layers[0]['weights'][winningNeuron] = new_weights.T
    def plotPerformance(self):


        ff = plt.figure(8)
        print(self.epoach_error)
        print(self.epoach_accuracy)
        x = np.arange(1,self.epoach+1)
        # for i in range(len(self.error_results)):
        #     self.error_results[i] = self.error_results[i][0,0]
        #
        #plt.plot(self.error_results)
        plt.title(r"Network Mean Square Error (lower is better) $\alpha$ %.4f  $\eta$ = %s" % (self.learning_rate, 30000))
        plt.plot(x,self.epoach_error, color='m',label="network error")
        plt.xlabel("Epochs iteration")
        plt.ylabel("Error mean %")
        plt.tight_layout()
        plt.legend()
        fff = plt.figure(23)
        plt.title(r"Network Accuracy $\alpha$ %.4f $\eta$ = %s" % (self.learning_rate, 30000))
        plt.xlabel("Epochs Iteration")
        plt.plot(x,self.epoach_accuracy, color='r', label="network accuracy")
        plt.legend()

        plt.tight_layout()
        plt.show()

        # fq = plt.figure(5)
        # plt.subplot(1,2,1)
        # plt.title(r"Network Mean Square Error (lower is better) $\alpha$ = %.4f" % self.learning_rate)
        # plt.plot(x,self.epoach_error, color='m',label="network error")
        # plt.legend()
        #
        # plt.subplot(1,2,2)
        # plt.title(r"Network Accuracy $\alpha$ = %.4f" % self.learning_rate)
        # plt.plot(x,self.epoach_accuracy, color='r', label="network accuracy")
        # plt.legend()
        #
        # plt.tight_layout()
        # plt.show()
    def predict(self, inputVector, targetVector):
        """
        Test the network's performance to perdict what the given handwritten vector is
        :param inputVector:
        :param targetVector:
        :return:
        """




        _t = np.matrix(self.targetStore[targetVector])# returns the target vector with hot encoding
        print("Encoding %s -> %s" % (targetVector, _t))
        #print("GOT : %s -> hot encoded to: %s" % ( targetVector, _t))
        _p = np.divide(np.matrix(inputVector),255)
        n_1 = np.matrix(-abs( np.linalg.norm( self.layers[0]['weights'].T - _p, axis=1) ) )
        a_1 = CompetitiveTrans(n_1.T)

        #second layer
        n_2 = np.dot(self.layers[1]['weights'], a_1)
        pur = PureLin()
        a_2 = pur(n_2.T)  # stand them up #10X 1

        print("exptected: %s |\n Got: %s" % (_t, a_2))
        return np.array_equal(_t, a_2)

def layer2Weight():
    x=np.array([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
           [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
    hot = 0
    m = []

    for i in range(1,51):
        m.append(x[hot])
        if not i == 0 and i % 5 == 0:
            hot += 1
    m = np.matrix(m)
    return m.T

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

    ################## Testing


    #images, labels = mndata.load_training()# loads only the training data
    images,labels = mndata.load_testing()
    imageFile = open('./Sources/images_Test.pickle','wb')
    pickle.dump(images,imageFile)
    labelsFile = open('./Sources/labels_Test.pickle','wb')
    pickle.dump(labels, labelsFile)

    imageFile.close()
    labelsFile.close()
    #############################
def loadPickleData():
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

    return imageVector, labelVector, imageVector_Test,labelVector_Test
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
        imagesVector, labelsVector, imageV_test, lablV_test = loadPickleData()
        print("file exist")
        return imagesVector,labelsVector, imageV_test,lablV_test
    else:
        pickleData()
        print("does not exist")

def testLVQNetwork():
    # make an instasnces of LVQNetwork
    # take the testing data,
    # run the test using the 'predict
    #function to test the network against what it gives and what wee expect


    imagesVector, labelsVector, imageVector_Test, labelsVector_Test = loadData() # load the data

    generateImageVector(imageVector_Test,labelsVector_Test)

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
        #for each testing pattern's index
        result = network.predict(imageVector_Test[pattern] , labelsVector_Test[pattern])
        #true if they are classified correctly false otherwise
        if result == True:
            #corectly classified
            results[labelsVector_Test[pattern]]['correctCount'] += 1 # increment
        else:
            results[labelsVector_Test[pattern]]['classifiedErrors'].append(labelsVector_Test[pattern])# record the patterns that were miss classified maybe get some insight on whats going on
        results[labelsVector_Test[pattern]]['testCount'] += 1 # increment number of times we've tested this pattern
        print(result)
    print("TOTAL TEST: %s" % len(imageVector_Test))
    # print the results in a

    x = np.arange(0,10)
    y = list()
    y_total = list()
    for i in sorted(results):
        y.append(results[i]['correctCount'])
        y_total.append(results[i]['testCount'])
    print(y)
    print(results)

    ff = plt.figure(6969)


    plt.bar(x + .25,y_total, label="Total Seen", color='gray')
    plt.bar(x,y, label="Num Correctly Classified")
    #plt.subplot(1,1,1)

    plt.xticks(x, ('0','1','2','3','4','5','6','7','8','9'))
    plt.ylabel("Number of correct classified")
    #plt.xlabel("Digit")
    plt.title("Performance for Number of Correctly Classified")
    plt.legend()

    plt.show()
def CompetitiveTrans(data):
    r = np.zeros_like(data)
    max = np.argmax(data) # get the index with the maximum value
    r[max] = 1.0
    return r # returns the maximum value the winning nurons index
# def compareTestAndTrain():
#     imagesVector, labelsVector, imageVector_Test, labelsVector_Test = loadData() # load the data
#     fig = plt.figure(685)
#     for row in range(1,5):
#         # 5 rows
#         for column in range(1,3):
#             # 2 columns
#             plt.subplot(4,2,)
#     for i in range(1,10):
#         plt.subplot(3,3,i)
#         train_p = np.matrix(imagesVector[])
def convertIntToTargetEncoded(number):
    imageLFile = open('./Sources/labels.pickle','rb')
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
    TRAINING = True

    if TRAINING:
        imagesVector, labelsVector, imageVector_Test, labelsVector_Test = loadData() # load the data
        net = Network(5, learning_rate=.5)


        #net.train(imagesVector[0:2000],labelsVector[0:2000])
        net.train1(imagesVector[0:1000], labelsVector[0:1000])


        net.predict(imagesVector[3000], labelsVector[3000])
        net.plotPerformance()

        f = open('./saveNetwork.pickle','wb')
        pickle.dump(net,f)
        f.close()
        plt.show()
    else:

        testLVQNetwork()
if __name__ == "__main__":
    main()







