import numpy as np

#sigmoid function
def nonlin(x,deriv = False):
    if(deriv == True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input dataset


X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])
Y = np.array([[0,0,1,1]]).T

np.random.seed(1)

syn0 = 2*np.random.random((3,1))-1

for iter in xrange(10000):
    l0 = X

    temp = np.dot(l0,syn0)

    l1 = nonlin(temp)


    #back propagation

    l1_error = Y - l1  #out-target

    l1_delta = l1_error*nonlin(l1,True)

    syn0 += np.dot(l0.T,l1_delta)

print l1