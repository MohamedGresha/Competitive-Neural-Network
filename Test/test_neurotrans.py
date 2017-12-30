# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
# =#| Author: Danny Ly MugenKlaus|RedKlouds
# =#| File:   test_neurotrans.py
# =#| Date:   12/8/2017
# =#|
# =#| Program Desc:
# =#|
# =#| Usage:
# =#|
# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
import numpy as np
import unittest
import neurotrans

class TestNeuroLib(unittest.TestCase):
    def setUp(self):
        self.test_input = np.array([1,2,3,0,5,3,-1])
        self.test_expt = np.array([1,2,34,5,9,3,-6])

    def test_Compet(self):

        compet = neurotrans.Compet()

        self.test_expt[5] = 9
        self.assertFalse(np.array_equal(compet(self.test_input), self.test_expt))

    def test_Prelin(self):
        purelin = neurotrans.PureLin()
        res = purelin(self.test_input)
        self.assertIsNot(res,self.test_input)
        self.assertTrue(np.array_equal(self.test_input,res))

    def test_Hardlim(self):
        hardlim = neurotrans.HardLim()
        res = hardlim(self.test_input)
        test_expt = np.array([1,1,1,1,1,1,0])
        self.assertIsNot(res, self.test_input)
        self.assertTrue(np.array_equal(res, test_expt))

    def test_HardLims(self):
        hardlims = neurotrans.HardLims()
        self.test_input[5] = -2
        res = hardlims(self.test_input)

        test_expt = np.array([1,1,1,1,-1,-1])
        self.assertFalse(np.array_equal(res,test_expt))

    def test_LogSigmoid(self):
        logsig = neurotrans.LogSig()

        res = logsig(self.test_input)
        self.assertIsNot(res, self.test_input)

        test_expt = np.array([ 0.73105858,  0.88079708,  0.95257413,  0.5,  0.99330715,
        0.95257413,  0.26894142])
        self.assertFalse(np.array_equal(res,test_expt))




if __name__ == "__main__":
    unittest.main()