#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   neurotrans.py
#=#| Date:   12/8/2017
#=#|
#=#| Program Desc: A object where all transfer functions are defined,
#=#| Helper class.
#=#|
#=#| Usage: Used when implementing layers with activation functions.
#=#|
#=#| Precondition: Requires Numpy library
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import numpy as np

class Compet:
    """
    Competitive Transfer function

    -returning the maximum value in the array
    Usage:
    p = [12,3,4,-1,5]

    compet = Compet()

    a = compet(p)

    a = [1,0,0,0,0]

    """
    def __call__(self, inputVector):
        r = np.zeros_like(inputVector)
        max = np.argmax(inputVector)  # get the index with the maximum value
        r[max] = 1.0
        return r  # returns the maximum value the winning nurons index


class PureLin:
    """
    Pure Linear Transfer function

    from neurotrans import PureLin

    Usage:
    p = [1,2,3,4,5]

    purelin = PureLin()

    a = purelin(p)

    a = [1,2,3,4,5]

    """
    def __call__(self, inputVector):
        return inputVector.copy()

class HardLim:
    """
    Hard limit Transfer Function

    Usage:
    p = [1,2,3,-5,-6,0]

    hardlim = HardLim()

    a = hardlim(p)

    a = [1,1,1,0,0,1]
    """

    def __call__(self, inputVector):
        r = np.zeros_like(inputVector)
        for i in range(len(inputVector)):
            if inputVector[i] >= 0:
                r[i] = 1
        return r

class HardLims:
    """
    Symmetric hard Limit Function

    Usage:
    from neurotrans import HardLims
    p = [-1,2,3,4,0,-2]

    hardlims = HardLims()

    a = hardlims(p)

    a = [-1,1,1,1,1,-1]

    """
    def __call__(self, inputVector):
        r = np.ones_like(inputVector)
        for i in range(len(inputVector)):
            if inputVector[i] < 0:
                r[i] = -1

        return r

class LogSig:
    """
    Log sigmoid Function

    Usage:
    from neutotrans import LogSig

    p = [12,3,4,5,6]

    logsig = LogSig()

    a = logsig(p)

    a = [ 0.99999386,  0.95257413,  0.98201379,  0.99330715,  0.99752738]

    """

    def __call__(self, inputvector):
        x = inputvector.copy()
        return 1 / (1 + np.exp( -x) )

    def derivative(self, inputVector):
        return inputVector * (1 - inputVector)