# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
# =#| Author: Danny Ly MugenKlaus|RedKlouds
# =#| File:   test_network.py
# =#| Date:   12/8/2017
# =#|
# =#| Program Desc:
# =#|
# =#| Usage:
# =#|
from unittest import TestCase
import unittest
import numpy as np
from CompetitiveNetwork import Network

# =#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
class TestNetwork(TestCase):

    def setUp(self):
        print("setupUp")
        self.id()
        self.network = Network()

    def test__dataDrivenInitalization(self):
        test_input = np.matrix([[1,2,3,4,5,6,7,8,.9],
                               [1,2,3,54,1,23,4,54,10],
                               [7,65,3,5,76,3,2,54,100]])
        test_target = np.array([0,1,2])
        #result = self.network._dataDrivenInitalization()
        self.assertRaises(Exception,self.network._dataDrivenInitalization,(123,50,test_input, test_target))
        self.assertRaises(Exception,self.network._dataDrivenInitalization,(-1,10,test_input, test_target))
        self.assertRaises(Exception,self.network._dataDrivenInitalization,(10,10,[],[]))
        result = self.network._dataDrivenInitalization(100,10,test_input,test_target)
    #     self.assertEqual(result.shape,(100,10) )

    def test__getHotEncoding(self):
        self.id()

        five = np.matrix([[0,0,0,0,0,1,0,0,0,0]])
        one = np.matrix([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
        two = np.matrix([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]])
        three = np.matrix([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
        nine = np.matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

        self.assertTrue(np.array_equal(five,self.network._getHotEncoding(5)) )

        self.assertTrue(np.array_equal(one, self.network._getHotEncoding(1)))
        self.assertTrue(np.array_equal(two, self.network._getHotEncoding(2)))
        self.assertTrue(np.array_equal(three, self.network._getHotEncoding(3)))
        self.assertTrue(np.array_equal(nine, self.network._getHotEncoding(9)))

        self.assertFalse( self.network._getHotEncoding(99))
        self.assertFalse(self.network._getHotEncoding(-1))

    def test__makeFinalClassLayer(self):
        result = self.network._makeFinalClassLayer(10, 500)
        self.assertEqual(result.shape, (10,500))
        self.assertRaises(Exception, self.network._makeFinalClassLayer, (12, 55))
        self.assertFalse(self.network._makeFinalClassLayer(-5, 10))


    def test__competitiveTrans(self):
        self.fail()

    def test__setUpNetwork(self):
        self.fail()

    def test_train(self):
        self.fail()

    def test__updateWeights(self):
        self.fail()

    def test_findDeadNeuron(self):
        self.fail()

    def test_plotPerformance(self):
        self.fail()

    def test_predict(self):
        self.fail()

    def test_savePrototypeWeights(self):
        self.fail()
if __name__ == "__main__":
    print("Main")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNetwork)
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()