#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
#=#| Author: Danny Ly MugenKlaus|RedKlouds
#=#| File:   CompetetiveBookProblem.py
#=#| Date:   12/2/2017
#=#|
#=#| Program Desc:
#=#|
#=#| Usage:
#=#|
#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#=#\|
from matplotlib import pyplot as plt

import numpy as np
p1 = np.matrix([[-.1961],[.9806]])
p2 = np.matrix([[.1961],[.9806]])
p3 = np.matrix([[.9806],[.1961]])
p4 = np.matrix([[.9806],[-.1961]])
p5 = np.matrix([[-.5812],[-.8137]])
p6 = np.matrix([[-.8137],[-.5812]])

p = [p1,p2,p3,p4,p5,p6]
x = []
y = []
for i in p:
    x.append(i[0][0,0])

for i in p:
    y.append(i[1][0,0])
plt.grid()
plt.scatter(x,y)
plt.axhline(linewidth=2,color='black')
plt.axvline(linewidth=2, color='black')
plt.xlim([-1.3,1.3])
plt.ylim([-1.3,1.3])

w_1 = np.matrix(np.random.uniform(low=0,high=1,size=(2,1)))
w_2 = np.matrix(np.random.uniform(low=0,high=1,size=(2,1)))
w_3 = np.matrix(np.random.uniform(low=0,high=1,size=(2,1)))

W = [[w_1.T],
               [w_2.T],
               [w_3.T]]
print(W)


### lets make some randomlized weights 'normalized with a length 1

#
# origin = [0],[0]
#
# #plt.quiver([0,0,0],[0,0,0],[1,-2,-4],[1,2,-7])

ww1 = np.matrix([[.7071],[-.7071]])
ww2 = np.matrix([[.7071],[.7071]])
ww3 = np.matrix([[-1.0000],[0.0000]])

W = np.array([[ww1.T],[ww2.T],[ww3.T]])
#multip[ly by vector 2

n = np.dot(W,p2)

print(n)
####### start afresh with arrays this time

ww1 = np.matrix([[.7071],[-.7071]])
ww2 = np.matrix([[.7071],[.7071]])
ww3 = np.matrix([[-1.0000],[0.0000]])

W = np.concatenate((ww1.T,ww2.T,ww3.T))
print(W)

# we got the winning neuron now lets adjust and updates its weights





def CompetitiveTrans(data):
    r = np.zeros_like(data)
    max = np.argmax(data) # get the index with the maximum value
    r[max] = 1.0
    return r # returns the maximum value the winning nurons index

n = np.dot(W,p2)

a = CompetitiveTrans(n)
print(a)
ww2 = ww2 + (.5 * (p2 - ww2) )

plt.scatter(ww2[0][0,0],ww2[1][0,0],edgecolors='r')
print(ww2[1][0,0])

plt.show()



